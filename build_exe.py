import PyInstaller.__main__
import os

# Ensure clean build
if os.path.exists("dist"):
    import shutil
    shutil.rmtree("dist")

PyInstaller.__main__.run([
    'main.py',
    '--name=AI_Humanizer_Pro',
    '--onefile',
    '--windowed', # Prevents console window from showing
    '--add-data=index.html;.', # Bundles the HTML file
    '--collect-all=transformers',
    '--collect-all=tiktoken',
    '--hidden-import=torch',
    '--hidden-import=tqdm',
    '--icon=NONE' # You can add an .ico file path here later
])

print("\nBuild Complete! Executable is in the 'dist' folder.")
print("Note: The model will download to the user's home directory (~/.cache/huggingface) on first run.")