```text
project_root/
├── notes/
│   └── architecture.md       # Ton Design Doc (Le cerveau du projet)
├── src/
│   ├── __init__.py
│   ├── core/                 # Les briques fondamentales (Maths pures)
│   │   ├── __init__.py
│   │   └── nurbs.py          # Ta classe NURBS (nn.Module)
│   ├── scene/                # L'assemblage (Data Container)
│   │   ├── __init__.py
│   │   ├── lights.py         # Tes PointsLights
│   │   └── composition.py    # Ta classe Scene (Liste d'objets + Lumières)
│   ├── rendering/            # La logique de transformation (Logic)
│   │   ├── __init__.py
│   │   └── renderer.py       # Prend une Scene + Camera -> Sort une Image
│   └── utils/                # Les outils transverses
│       ├── __init__.py
│       └── math_ops.py       # Fonctions statiques (ex: transformations géométriques)
├── tests/                    # Le Miroir de src
│   ├── conftest.py           # Les Fixtures (Données de test)
│   ├── core/
│   │   └── test_nurbs.py
│   └── rendering/
│       └── test_renderer.py
├── pyproject.toml            # Dépendances et config
├── environment.yml 
└── README.md
```


Class :
    Docstring 
    __init__
    @property
    forward
    methodes publiques
    méthodes privés 


# Architecture du Projet NURBS-Lighter

## Objectif
Optimiser une surface NURBS via PyTorch avec gestion de la topologie (clamping) et rendu differentiable.

## 1. Core : La Classe NURBS
**Fichier :** `src/core/nurbs.py`
**Input :** Coordonnées paramétriques (u, v).
**Parameters :**
- `control_points`: Float[Tensor, "num_cp 4"] (Homogène w pour rationalité)
- `knots_u`, `knots_v`: Float[Tensor, "num_knots"]
  **Méthodes Clés :**
- `_compute_basis`: TODO (Expliquer en une phrase ce qu'elle fait)
- `_nurbs_divide`: Gère la division 0/0.
- `evaluate`: Calcule P(u,v) et la Normale N(u,v).

## 2. Rendering : Le Pipeline
**Fichier :** `src/rendering/renderer.py`
**Input :** Une instance de `Scene` et des `Cameras`.
**Stratégie de Rendu :** TODO (Est-ce que tu fais du Rasterization via Kaolin ou du Point Cloud Rendering ?)

## 3. Conventions
- Types : Float32 partout.
- Tests : Pytest avec Fixtures dans `tests/conftest.py`.
