"""Silence robosuite's verbose [INFO] logs.

Robosuite affiche par défaut des lignes type "Loading controller configuration from..."
à chaque env.reset(), ce qui pollue les logs. Cette fonction réduit la verbosité
à WARNING — on garde toujours les warnings et erreurs.

Usage :
    from src.quiet_robosuite import silence_robosuite
    silence_robosuite()
    import robosuite as rs  # ou après l'import, peu importe
"""

import logging


def silence_robosuite(level: str = "WARNING"):
    """Reduce robosuite logging verbosity. Defaults to WARNING."""
    # Loggers Python connus de robosuite (couvre robosuite_logs et variantes)
    for name in list(logging.root.manager.loggerDict.keys()):
        if "robosuite" in name.lower() or "imageio" in name.lower():
            logging.getLogger(name).setLevel(getattr(logging, level, logging.WARNING))

    # Macros internes de robosuite (qui font des prints stdout en plus du logging)
    try:
        import robosuite.macros as macros
        macros.LOGGING_LEVEL = level
    except Exception:
        pass

    # Force le niveau aussi sur le logger racine "robosuite_logs" même si pas encore créé
    logging.getLogger("robosuite_logs").setLevel(getattr(logging, level, logging.WARNING))
