# .devcontainer — Cómo usar y configurar en VS Code ✅

## Descripción
Este documento explica **cómo instalar**, **abrir** y **personalizar** el entorno de desarrollo definido en `.devcontainer` usando Visual Studio Code.

---

## Requisitos previos 🔧
- Docker (o Docker Desktop) instalado y en ejecución — [Docker Desktop](https://www.docker.com/products/docker-desktop/).
- Visual Studio Code actualizado — [Visual Studio Code](https://code.visualstudio.com/).
- Extensión **Dev Containers** instalada: `ms-vscode-remote.vscode-dev-containers` — [Descargar extensión](https://aka.ms/vscode-remote/download/extension).

---

## Abrir el proyecto dentro del contenedor (pasos rápidos) ⚡
1. Abre la carpeta del repositorio en VS Code (la que contiene `.devcontainer/`).
2. Abre la paleta de comandos (Ctrl+Shift+P) y ejecuta:
   - `Dev Containers: Reopen in Container`
3. VS Code construirá o descargará la imagen y abrirá el proyecto dentro del contenedor.

---

## Comandos útiles en VS Code ✅
- `Dev Containers: Rebuild Container` — Reconstruye el contenedor desde cero.
- `Dev Containers: Reopen in Container` — Reabre la carpeta dentro del contenedor.
- `Dev Containers: Attach to Running Container` — Se conecta a un contenedor ya en ejecución.
- Ver salida del proceso: Panel *Output* → seleccionar *Dev Containers*.

---

## Qué hace el `.devcontainer/devcontainer.json` en este repositorio 🔍
- Usa la imagen `python:3.13.7-bookworm`.
- Añade features (utilidades comunes y Python).
- Ejecuta `updateContentCommand` para instalar paquetes del sistema y `pip3 install --user -r requirements.txt`.
- Instala extensiones listadas en `customizations.vscode.extensions`.

---

## Solución de problemas ⚠️
- Si la construcción falla: revisa la salida en *Dev Containers* y confirma que Docker está activo.
- Si las extensiones no se instalan: reconstruye el contenedor (`Rebuild Container`).
- Para ejecutar comandos manuales dentro del contenedor: abre *Terminal → New Terminal* en VS Code.

---

## Verificación rápida ✅
1. Tras abrir el contenedor, abre un terminal integrado y ejecuta:
   - `python --version` (debería corresponder a la imagen)
   - `pip3 list` (ver paquetes instalados)
2. Si tu flujo usa notebooks, abre la carpeta `notebooks/` y ejecuta las celdas con el kernel del entorno.