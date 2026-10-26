import os

# tests run alongside other work: few threads, low priority (applied when the package is imported)
os.environ.setdefault("SI_THREADS", "2")
os.environ.setdefault("SI_PRIORITY", "low")
