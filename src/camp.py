"""CAMP-lite — mémoire compressée de l'historique d'ACTIONS pour une Diffusion Policy.

D'après *Remember what you did? Learning Behavioral Memories for Partially Observable
Object Manipulation* (Wang, Yeom, Cao, Zhi, Shinde, Yip — arXiv 2606.21188, UCSD ARCLAB).
Pas de code publié à ce jour (site projet robo-camp.github.io : aucun dépôt) → ré-implémenté
d'après le papier. (Renvoyait à docs/recherche/MEMOIRE.md : document introuvable au
2026-09-13, jamais versionné.)

IDÉE — l'historique des actions du robot est un signal auto-supervisé riche : il encode
« ce que j'ai déjà tenté », donc les échecs, les reprises, la progression. On entraîne un
petit LSTM à RECONSTRUIRE sa propre trajectoire d'actions passée ; s'il en est capable,
c'est que son état caché la contient. Cet état, compressé et quantifié en 32 nombres, est
ensuite donné à la policy comme une entrée de plus.

CE QUE LE PAPIER FAIT vs CE QU'ON FAIT (et pourquoi)
----------------------------------------------------
Le papier fait **pré-entraînement PUIS FINETUNING CONJOINT** (LSTM à lr × α=0.1, encodeur
visuel partagé avec la tête d'action). Le finetuning conjoint oblige à dérouler le LSTM
pendant l'entraînement de la policy → coût mémoire et plomberie dans la boucle
d'entraînement. C'est exactement ce que la contrainte C1 de Sam interdit
(« l'entraînement ne doit pas être significativement plus long »).

D'où DEUX variantes, à faire dans cet ordre :

  variante B — « CAMP-lite », celle qu'on implémente ici. Le module mémoire est
    pré-entraîné puis **GELÉ DÉFINITIVEMENT**. On précalcule m_t pour toutes les frames et
    on l'écrit dans `observation.state` d'un dataset dérivé. L'entraînement de la policy est
    alors **rigoureusement la boucle actuelle**, sans une ligne de LeRobot modifiée et sans
    surcoût. C1 satisfaite par construction.

  variante A — fidèle au papier (warm-up gelé puis finetuning conjoint α=0.1). À ne tenter
    QUE si B plafonne : on saura alors que le finetuning conjoint compte vraiment, et on
    décidera en connaissance de cause s'il mérite son coût.

⚠️ Ne pas présenter B comme « CAMP ». C'est CAMP amputé de son finetuning conjoint. Un
échec de B ne réfute pas CAMP.

VISION OU PAS
-------------
Le papier alimente le LSTM avec [features visuelles, proprio, action précédente]. On expose
`use_vision`, mais le défaut est **sans vision** — et ce n'est pas qu'une simplification :
c'est l'hypothèse pure de CAMP, testée nue. Si le bras est descendu vers un point X puis a
échoué, **la trajectoire passée encode X**, c'est-à-dire où il croyait que la cible était.
Autrement dit, sur une tâche M(1), *ton propre mouvement passé encode ta croyance sur le
monde* — la position de la canette est déjà implicitement là, sans une seule image.
Sans vision, le module devient trivialement léger et indépendant de l'encodeur de la policy
(qui n'existe pas encore au moment du pré-entraînement).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------- DCT
def dct_matrix(length, n_coef, dtype=torch.float32):
    """Matrice DCT-II orthonormée tronquée : (n_coef, length).

    Projette une trajectoire de `length` pas sur ses `n_coef` premières fréquences.
    Coefficient 0 = la moyenne (« où était le bras en gros ») ; coefficients bas = les
    ondulations lentes (le geste) ; coefficients hauts = les micro-corrections et le bruit.
    Même principe que le JPEG : on garde les basses fréquences, le geste reste reconnaissable.
    """
    n = np.arange(length)[None, :]
    k = np.arange(n_coef)[:, None]
    m = np.cos(np.pi * (2 * n + 1) * k / (2 * length))
    m *= np.sqrt(2.0 / length)
    m[0] *= 1 / np.sqrt(2.0)
    return torch.tensor(m, dtype=dtype)


def freq_weights(n_coef, gamma=3.0, dtype=torch.float32):
    """Pondération fréquentielle w_k = exp(-γ k / (K-1)) du papier.

    Écrase les hautes fréquences dans la perte. Deux effets, tous deux voulus :
      1. le LSTM retient la FORME du geste et peut oublier le détail ;
      2. ⭐ anti-copycat — l'action précédente a_{t-1} est un détail de très haute
         fréquence de la trajectoire ; la dé-pondérer décourage mécaniquement le réseau
         de mémoriser « ce que je viens de faire » au profit de « l'allure de ce que j'ai
         fait ». C'est la première des deux barrières (l'autre est la quantification).
    """
    k = torch.arange(n_coef, dtype=dtype)
    return torch.exp(-gamma * k / max(n_coef - 1, 1))


# ------------------------------------------------------------------- quantifieur vectoriel
class VectorQuantizer(nn.Module):
    """VQ-VAE standard (straight-through + commitment).

    ⭐ C'est la barrière anti-copycat DURE. Le message qui atteint le dénoiseur ne peut
    prendre que `codebook_size` valeurs distinctes — 128 codes = 7 bits par instant. Il est
    *mathématiquement impossible* d'y cacher une action 7D en clair. Là où la pondération
    fréquentielle décourage le copycat, la quantification le rend infaisable.

    Effet secondaire recherché : les codes finissent par correspondre à des « types de
    moment » (approche, saisie, échec, reprise). Une mémoire discrète ne dérive pas
    doucement hors distribution, elle saute d'une catégorie connue à une autre.
    """

    def __init__(self, dim=32, codebook_size=128, beta=0.25):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.beta = beta

    def forward(self, z):
        """z : (..., dim) -> (z_q, loss_vq, indices). z_q est droit-passant (gradient = celui de z)."""
        flat = z.reshape(-1, z.shape[-1])
        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ self.codebook.weight.t()
             + self.codebook.weight.pow(2).sum(1)[None, :])
        idx = d.argmin(1)
        zq = self.codebook(idx).view_as(z)
        loss = F.mse_loss(zq, z.detach()) + self.beta * F.mse_loss(z, zq.detach())
        zq = z + (zq - z).detach()          # straight-through
        return zq, loss, idx.view(z.shape[:-1])

    @torch.no_grad()
    def perplexity(self, idx):
        """Nombre EFFECTIF de codes utilisés. À surveiller pendant le pré-entraînement :
        le mode d'échec classique du VQ est le *collapse* — tout tombe sur 1-2 codes et la
        mémoire ne transporte plus rien. Une perplexité qui s'effondre vers 1 = mémoire
        morte, et la policy en aval ne le dira pas (elle apprendra juste à l'ignorer)."""
        counts = torch.bincount(idx.flatten(), minlength=self.codebook.num_embeddings).float()
        p = counts / counts.sum().clamp(min=1)
        return float(torch.exp(-(p * (p + 1e-10).log()).sum()))


# --------------------------------------------------------------------------- module mémoire
class CampMemory(nn.Module):
    """LSTM 2 couches (h=64) -> tête DCT (perte) + projection quantifiée (le code mémoire).

    Deux sorties depuis le même état caché :
      * `coef`  : g_φ(h_t), les K coefficients DCT par dimension d'action — sert UNIQUEMENT
                  à la perte de pré-entraînement, jeté ensuite ;
      * `m_t`   : proj(h_t) quantifié, `mem_dim` nombres — c'est ce qui part vers la policy.
    """

    def __init__(self, action_dim, state_dim, *, n_coef=32, hidden=64, layers=2,
                 mem_dim=32, codebook_size=128, vision_dim=0):
        super().__init__()
        self.action_dim, self.state_dim, self.n_coef = action_dim, state_dim, n_coef
        self.vision_dim = vision_dim
        in_dim = state_dim + action_dim + vision_dim
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, n_coef * action_dim)
        self.proj = nn.Linear(hidden, mem_dim)
        self.vq = VectorQuantizer(mem_dim, codebook_size)

    def forward(self, state, prev_action, vision=None, hx=None):
        """state (B,T,S), prev_action (B,T,A), vision (B,T,V)|None -> coef, m, loss_vq, hx.

        `hx` permet de reprendre l'état caché d'un segment au suivant (déroulement par
        morceaux) et, à l'inférence, de faire avancer la mémoire pas à pas.
        """
        parts = [state, prev_action] if vision is None else [state, prev_action, vision]
        h, hx = self.lstm(torch.cat(parts, dim=-1), hx)
        coef = self.head(h).view(*h.shape[:2], self.n_coef, self.action_dim)
        m, loss_vq, idx = self.vq(self.proj(h))
        return coef, m, loss_vq, idx, hx


# --------------------------------------------------------------------------------- pertes
def past_window(actions, t_idx, length):
    """Fenêtre d'actions passées a_{t-L:t-1} pour chaque t, padding par répétition de a_0.

    Padding par répétition et non par des zéros : avant le début de l'épisode le bras était
    IMMOBILE, pas à l'origine du repère. Des zéros injecteraient un faux mouvement dans la
    DCT.
    """
    B, T, A = actions.shape
    idx = t_idx[:, None] - length + torch.arange(length, device=actions.device)[None, :]
    idx = idx.clamp(min=0)                                   # (T, L)
    return actions[:, idx, :]                                # (B, T, L, A)


def camp_losses(coef_pred, actions, dct, w, *, lam_cons=1.0, n_shifts=(1, 2, 4)):
    """Perte de pré-entraînement = reconstruction DCT pondérée + cohérence temporelle.

    ① reconstruction : les coefficients prédits doivent égaler ceux de la vraie trajectoire
       passée, pondérés en fréquence (basses fréquences prioritaires).
    ② cohérence temporelle : les résumés produits à t et t+N décrivent des fenêtres qui se
       RECOUVRENT ; on exige qu'ils s'accordent sur la partie commune.
       ⚠️ Ce n'est pas cosmétique : cette mémoire devient une ENTRÉE du dénoiseur, et une
       entrée qui saute produit des actions qui sautent — c'est-à-dire des saccades, le
       problème déjà connu sur le vrai bras (note atomman §5). Cette perte est ce qui rend
       la mémoire utilisable comme conditionnement.
    """
    B, T, K, A = coef_pred.shape
    L = dct.shape[1]
    t_idx = torch.arange(T, device=actions.device)
    target = past_window(actions, t_idx, L)                       # (B,T,L,A)
    coef_true = torch.einsum("kl,btla->btka", dct, target)        # (B,T,K,A)

    l_rec = (w[None, None, :, None] * (coef_pred - coef_true).pow(2)).mean()

    recon = torch.einsum("kl,btka->btla", dct, coef_pred)         # trajectoires reconstruites
    l_cons = coef_pred.new_zeros(())
    for n in n_shifts:
        if T > n and L > n:
            l_cons = l_cons + F.mse_loss(recon[:, :-n, n:, :], recon[:, n:, :L - n, :])
    l_cons = l_cons / max(len(n_shifts), 1)
    return l_rec + lam_cons * l_cons, {"rec": float(l_rec.detach()), "cons": float(l_cons.detach())}


# ------------------------------------------------------- phase 2 : injection dans la policy
@torch.no_grad()
def precompute_memory(dataset, model, n_obs_steps, device, batch_episodes=8):
    """Déroule le module mémoire sur TOUT le dataset et empile les codes par pas d'observation.

    Retourne `M` de forme (num_frames_total, n_obs_steps, mem_dim), indexable directement
    par `batch["index"]` (l'index GLOBAL de la frame courante fourni par LeRobot).

    Pourquoi pré-empiler plutôt qu'indexer `M[index-1]` à la volée : au premier pas d'un
    épisode, `index-1` pointe sur la DERNIÈRE frame de l'épisode PRÉCÉDENT. On clampe donc
    ici, une fois, là où les frontières d'épisode sont connues — plutôt que de laisser une
    fuite silencieuse d'un épisode à l'autre dans la boucle d'entraînement.

    ⚠️ Le module est GELÉ (variante B, cf. en-tête). On peut donc tout précalculer une fois :
    l'entraînement de la policy reste ensuite RIGOUREUSEMENT la boucle habituelle.
    """
    model.eval().to(device)
    hf = dataset.hf_dataset.select_columns(["observation.state", "action", "episode_index", "index"])
    ep_idx = np.asarray(hf["episode_index"])
    gidx = np.asarray(hf["index"])
    states = np.asarray(hf["observation.state"], dtype=np.float32)
    actions = np.asarray(hf["action"], dtype=np.float32)

    mem_dim = model.proj.out_features
    M = torch.zeros(int(gidx.max()) + 1, n_obs_steps, mem_dim)
    for ep in np.unique(ep_idx):
        sel = np.where(ep_idx == ep)[0]
        p = torch.tensor(states[sel], device=device)[None]          # (1,T,S)
        a = torch.tensor(actions[sel], device=device)[None]         # (1,T,A)
        prev = torch.cat([torch.zeros_like(a[:, :1]), a[:, :-1]], dim=1)  # a_{t-1}, ZÉRO au 1er pas
        _, m, _, _, _ = model(p, prev)                              # (1,T,mem)
        m = m[0].cpu()
        # empilage des n_obs_steps derniers codes, clampé au DÉBUT de l'épisode
        T = m.shape[0]
        offs = torch.arange(-(n_obs_steps - 1), 1)                  # ex. [-1, 0]
        rows = (torch.arange(T)[:, None] + offs[None, :]).clamp(min=0)
        M[torch.tensor(gidx[sel])] = m[rows]
    return M


# ------------------------------- variante A : finetuning CONJOINT (fidèle au papier) -------
class EpisodeBank:
    """Séquences (proprio, action) de tout le dataset, en RAM, pour dérouler le LSTM À LA VOLÉE.

    POURQUOI c'est peu coûteux — et pourquoi j'avais tort de reculer devant la BPTT.
    Le module mémoire ne consomme AUCUNE image : ses entrées sont la proprioception et
    l'action précédente. Dérouler un LSTM de 64 dims sur un épisode entier (~150 pas) pour
    32 échantillons, c'est ~4500 pas de LSTM minuscule — quelques millisecondes, à comparer
    aux ~600 ms du dénoiseur. Le déroulement complet est donc ABORDABLE, et le gradient peut
    remonter jusqu'au LSTM : c'est ce que fait le papier (`supervising over the full episode
    length L`), et ce que la variante B (module gelé) ne fait pas.

    Tout le dataset tient en RAM : 23 207 frames × 16 floats ≈ 1,5 Mo.
    """

    def __init__(self, dataset, device):
        hf = dataset.hf_dataset.select_columns(
            ["observation.state", "action", "episode_index", "frame_index"])
        ep = np.asarray(hf["episode_index"]); fr = np.asarray(hf["frame_index"])
        st = np.asarray(hf["observation.state"], dtype=np.float32)
        ac = np.asarray(hf["action"], dtype=np.float32)
        self.ids = np.unique(ep)
        self.row = {int(e): i for i, e in enumerate(self.ids)}
        self.maxlen = int(max((ep == e).sum() for e in self.ids))
        S, A = st.shape[1], ac.shape[1]
        P = np.zeros((len(self.ids), self.maxlen, S), np.float32)
        Q = np.zeros((len(self.ids), self.maxlen, A), np.float32)
        for e in self.ids:
            sel = np.where(ep == e)[0][np.argsort(fr[ep == e])]
            r, n = self.row[int(e)], len(sel)
            P[r, :n] = st[sel]
            Q[r, 1:n] = ac[sel][:-1]        # action PRÉCÉDENTE, zéro au 1er pas
        self.P = torch.tensor(P, device=device)
        self.Q = torch.tensor(Q, device=device)

    def memory_for(self, model, episode_index, frame_index):
        """Déroule le LSTM depuis le DÉBUT de chaque épisode et renvoie m_t à `frame_index`.

        Le gradient traverse tout le déroulement -> le LSTM est réellement finetuné.
        """
        rows = torch.tensor([self.row[int(e)] for e in episode_index.tolist()],
                            device=self.P.device)
        _, m, loss_vq, _, _ = model(self.P[rows], self.Q[rows])      # (B, maxlen, mem)
        idx = frame_index.to(m.device).long().clamp(0, m.shape[1] - 1)
        return m[torch.arange(m.shape[0], device=m.device), idx], loss_vq
