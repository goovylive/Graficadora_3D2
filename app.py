"""
Backend FastAPI para computar mallas 3D de superficies implícitas f(x,y,z,t)=0
usando Marching Cubes (scikit-image) y evaluación segura con eval.
"""
import re
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from skimage import measure

app = FastAPI(
    title="Implicit Surface Mesh API",
    description="API para calcular vértices y caras de superficies implícitas 4D f(x,y,z,t)=0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Modelos de request/response ---

class Ranges(BaseModel):
    xMin: float = Field(..., description="Límite inferior en X")
    xMax: float = Field(..., description="Límite superior en X")
    yMin: float = Field(..., description="Límite inferior en Y")
    yMax: float = Field(..., description="Límite superior en Y")
    zMin: float = Field(..., description="Límite inferior en Z")
    zMax: float = Field(..., description="Límite superior en Z")


class ComputeMeshRequest(BaseModel):
    equation: str = Field(..., description="Ecuación f(x,y,z,t)=0 en sintaxis numexpr")
    ranges: Ranges = Field(..., description="Rangos espaciales")
    t: float = Field(..., description="Valor del parámetro temporal t")
    resolution: int = Field(18, ge=6, le=50, description="Resolución de la malla (ej: 18 → grid 18³)")


# Caracteres/funciones permitidos para sanitización básica
_UNSAFE_PATTERNS = re.compile(
    r'["\'\\]|__|import|eval|exec|open|file|getattr|setattr|globals|locals|breakpoint|compile'
)


def _sanitize_equation(equation: str) -> str:
    """Normaliza y sanitiza la ecuación para evaluación segura."""
    if not equation or not equation.strip():
        raise ValueError("La ecuación no puede estar vacía")
    eq = equation.strip()
    if _UNSAFE_PATTERNS.search(eq):
        raise ValueError("La ecuación contiene caracteres o términos no permitidos")
    eq = eq.replace("^", "**")
    eq = re.sub(r"\bpow\s*\(", "power(", eq, flags=re.IGNORECASE)
    eq = re.sub(r"\bsqrt\s*\(", "safe_sqrt(", eq, flags=re.IGNORECASE)
    return eq


def _evaluate_equation(equation: str, X, Y, Z, t):
    eq = _sanitize_equation(equation)

    def safe_sqrt(arr):
        return np.sqrt(np.maximum(arr, 0.0))

    local_dict = {
        "x": X,
        "y": Y,
        "z": Z,
        "t": t,
        "sin": np.sin,
        "cos": np.cos,
        "sqrt": safe_sqrt,
        "safe_sqrt": safe_sqrt,
        "exp": np.exp,
        "log": np.log,
        "abs": np.abs,
        "power": np.power,
        "tan": np.tan,
        "pi": np.pi,
        "e": np.e,
    }

    try:
        namespace = {**local_dict}
        result = eval(eq, {"__builtins__": {}}, namespace)
        if np.isscalar(result):
            result = np.full_like(X, result)
    except Exception as e:
        raise ValueError(f"Error al evaluar la ecuación: {e}") from e

    if not isinstance(result, np.ndarray) or result.shape != X.shape:
        raise ValueError("La ecuación debe producir un array 3D")

    result = np.nan_to_num(result, nan=1e10, posinf=1e10, neginf=-1e10)

    return np.asarray(result, dtype=np.float64)


def _marching_cubes_mesh(
    equation: str,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    z_min: float, z_max: float,
    t: float,
    resolution: int,
) -> tuple[list[float], list[int]]:
    """
    Construye la malla 3D de la superficie implícita f(x,y,z,t)=0.
    Retorna (vertices_flat, faces_flat).
    - vertices_flat: [x1,y1,z1, x2,y2,z2, ...]
    - faces_flat: [i0,j0,k0, i1,j1,k1, ...] índices de triángulos.
    """
    n = resolution
    x = np.linspace(x_min, x_max, n)
    y = np.linspace(y_min, y_max, n)
    z = np.linspace(z_min, z_max, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    volume = _evaluate_equation(equation, X, Y, Z, t)

    spacing = (
        (x_max - x_min) / (n - 1) if n > 1 else 1.0,
        (y_max - y_min) / (n - 1) if n > 1 else 1.0,
        (z_max - z_min) / (n - 1) if n > 1 else 1.0,
    )
    origin = (x_min, y_min, z_min)

    try:
        result = measure.marching_cubes(
            volume, level=0.0, spacing=spacing, gradient_direction="descent"
        )
        verts = result[0]
        faces = result[1]
    except ValueError as e:
        if "surface must cross the volume" in str(e).lower() or "no surface" in str(e).lower():
            return [], []
        raise

    # Pasar a coordenadas físicas (skimage devuelve en índice * spacing; sumamos origin)
    verts_physical = verts * np.array(spacing) + np.array(origin)

    # vertices: [x1,y1,z1, x2,y2,z2, ...]
    vertices_flat = verts_physical.astype(np.float64).ravel().tolist()

    # faces: [i0,j0,k0, i1,j1,k1, ...]
    faces_flat = faces.astype(np.int32).ravel().tolist()

    return vertices_flat, faces_flat


@app.post("/compute-mesh")
def compute_mesh(req: ComputeMeshRequest):
    """
    Calcula la malla (vértices y caras) de la superficie implícita f(x,y,z,t)=0
    para el valor de t y los rangos dados.
    """
    r = req.ranges
    try:
        vertices, faces = _marching_cubes_mesh(
            equation=req.equation,
            x_min=r.xMin, x_max=r.xMax,
            y_min=r.yMin, y_max=r.yMax,
            z_min=r.zMin, z_max=r.zMax,
            t=req.t,
            resolution=req.resolution,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"vertices": vertices, "faces": faces}


@app.get("/health")
def health():
    return {"status": "ok"}
