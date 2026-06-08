🤝 Contribuyendo a TruePixel

¡Primero que nada, gracias por tomarte el tiempo de contribuir a la transparencia y la verdad digital! TruePixel es un esfuerzo comunitario enfocado en democratizar las herramientas de verificación forense digital.

Para mantener la calidad, legibilidad y el rigor científico del código, te pedimos amablemente que sigas estas pautas.

👉 English version of this document

🛠️ Flujo de Trabajo para Pull Requests

Haz un Fork del repositorio a tu cuenta personal.

Crea una rama de características (git checkout -b feature/nueva-mejora-espectral).

Escribe y documenta tu código. Asegúrate de mantener la compatibilidad con PyTorch y de no romper los procesos modulares.

Prueba localmente: Asegúrate de que train.py (usando tu dataset local) y api.py funcionen sin errores en tu entorno virtual local.

Realiza un Commit claro y descriptivo (git commit -m 'feat: optimiza la rama de Fourier usando ventanas espectrales').

Sube tus cambios a tu rama (git push origin feature/nueva-mejora-espectral).

Abre un Pull Request (PR) hacia la rama principal (main) de TruePixel explicando con detalle el cambio realizado.

📏 Estilo de Código y Estándares

Para garantizar que ingenieros de todo el mundo puedan colaborar sin fricción, buscamos seguir estas normas de programación:

Pythonic Code: Adherencia general a la guía de estilo PEP 8.

Modularidad en Machine Learning: No mezcles procesamiento de datos local con la arquitectura del modelo. Mantén model.py, fourier.py y dataset.py como módulos puros y limpios.

Comentarios de Código: Escribe comentarios claros en funciones matemáticas críticas para explicar el por qué de las operaciones (especialmente en transformaciones de espectro y normalizaciones).

Pesos de Entrenamiento: Por favor, no subas archivos de pesos .pth o .pt pesados en tus Pull Requests. Los pesos optimizados o entrenados en masa deben ser distribuidos mediante las secciones de "Releases" de GitHub o plataformas de almacenamiento de modelos como Hugging Face.

🐞 Reportar Errores o Proponer Ideas

Si encontraste un fallo en la detección de un modelo generador nuevo (como nuevas versiones de Midjourney) o tienes una sugerencia matemática para mejorar el motor:

Dirígete a la pestaña de Issues en el repositorio.

Utiliza una plantilla descriptiva: especifica el comportamiento esperado, el comportamiento observado y proporciona una imagen de ejemplo si es posible.

¡Agradecemos enormemente tu dedicación por construir un internet libre de manipulación visual!