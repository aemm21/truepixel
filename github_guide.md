🚀 Cómo subir TruePixel a GitHub

Sigue estos pasos en la terminal de VS Code para publicar tu proyecto al mundo. Asegúrate de estar en la carpeta raíz de tu proyecto.

Paso 1: Inicializar Git

Este comando convierte tu carpeta normal en un repositorio rastreado por Git.

git init


Paso 2: Añadir los archivos

El punto . significa "añade todos los archivos de esta carpeta" (respetando lo que pusimos en .gitignore).

git add .


Paso 3: Guardar el estado (Commit)

Crea una "fotografía" de tu código actual con un mensaje descriptivo.

git commit -m "Lanzamiento inicial de TruePixel Core y API"


Paso 4: Conectar con GitHub

Ve a GitHub.com e inicia sesión.

Haz clic en el botón verde "New" (Nuevo repositorio).

Ponle de nombre truepixel, una descripción, y déjalo como Public. NO marques la casilla de añadir un README (ya lo tienes).

Haz clic en Create repository.

GitHub te mostrará una pantalla con comandos. Copia las tres últimas líneas que aparecen en la sección "…or push an existing repository from the command line" y pégalas en tu terminal. Se verán parecidas a esto:

git branch -M main
git remote add origin [https://github.com/TU-USUARIO/truepixel.git](https://github.com/TU-USUARIO/truepixel.git)
git push -u origin main


¡Listo! Si recargas la página de GitHub, tu código ya será público y estará disponible para cualquier desarrollador en el planeta.