@echo off
setlocal enabledelayedexpansion

REM Create raw data directory
if not exist "data\raw" mkdir "data\raw"

REM WIDER FACE (detection)
echo Download WIDER FACE manually and place in data\raw\widerface\

REM VGGFace2
echo Download VGGFace2 following instructions: https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/

REM MaskedFace-Net
echo Downloading MaskedFace-Net (if possible)...
if not exist "data\raw\maskedface" mkdir "data\raw\maskedface"
powershell -Command "try { Invoke-WebRequest -Uri 'https://github.com/...' -OutFile 'data/raw/maskedface/maskedface-net.zip' } catch { Write-Host 'Please download MaskedFace-Net manually if download fails.' }"

REM RMFRD
echo Place RMFRD dataset under data\raw\rmfrd\

REM IJB-B / IJB-C
echo Download IJB-B/IJB-C via IARPA Janus license; place in data\raw\ijb\

echo Datasets: Please follow the printed instructions to place datasets in data\raw\
