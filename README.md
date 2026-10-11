# Skills Matrix Builder

Aplicación en Streamlit que usa IA (API de Claude) para que RR.HH. construya una matriz de habilidades por rol con niveles de madurez por seniority, compare niveles (base del plan de carrera) y genere descripciones de puesto. La IA propone; RR.HH. revisa, edita y aprueba cada paso.

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sin clave de API, activa **Modo demo** en la barra lateral y pulsa **Cargar ejemplo completo**.

## Desplegar en Streamlit Community Cloud

1. Sube a un repositorio de GitHub: `app.py`, `requirements.txt`, `README.md`, `.gitignore` y `.streamlit/secrets.toml.example`.
2. En Streamlit Community Cloud: **Create app** → repositorio → rama `main` → archivo `app.py`.
3. En **Advanced settings → Secrets** pega `ANTHROPIC_API_KEY = "tu-clave"`.
4. **Deploy**.

La clave nunca va en el código ni en GitHub. Si no se configura, la app pide la clave en la barra lateral o se puede usar el modo demo.

## Notas

- Streamlit Community Cloud no guarda datos: exporta la matriz en JSON para retomarla e impórtala desde la barra lateral.
- La app trabaja sobre roles, no sobre personas, y no puntúa el desempeño de empleados.
