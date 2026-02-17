---
title: Implicit Surface Mesh API
emoji: 🌀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Implicit Surface Mesh API

Backend API para calcular superficies implícitas 4D usando Marching Cubes.

## Endpoints:
- POST /compute-mesh - Calcula malla 3D
- GET /health - Health check

## Uso:
```bash
curl -X POST https://[TU-SPACE].hf.space/compute-mesh \
  -H "Content-Type: application/json" \
  -d '{
    "equation": "x*x + y*y + z*z - 4",
    "ranges": {"xMin":-3,"xMax":3,"yMin":-3,"yMax":3,"zMin":-3,"zMax":3},
    "t": 0,
    "resolution": 18
  }'
```
