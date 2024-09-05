# Execute the following command in the bfqcircuits directory
# to build the sphinx documentation: powershell.exe -File docs\build.ps1
# You have to install the packages specified in the requirements.txt as well as pandoc
# or copy the pandoc.exe to the bfqcircuits folder

# Build html
sphinx-build -M html docs docs/build

.\docs\build\html\index.html