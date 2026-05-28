# Párhuzamos Eszközök Programozása

## A feladat célja

A feladat célja egy egyszerű folyadékszimuláció elkészítése, majd annak hatékonysági összehasonlítása CPU-s és OpenCL-es megvalósítás esetén.

## Projektstruktúra

A megvalósításhoz szükséges programfájlok a "FluidSim" mappában találhatók.

## Felhasznált könyvtárak és eszközök

* OpenCL
* FFmpeg
* C fordító

## Fordítás és futtatás

### build_compare.bat

Lefuttatja a CPU-s és az OpenCL-es változatot is, majd röviden kiértékeli az eredményeket.

### build_cpu.bat

A program CPU-s változatát fordítja és futtatja.

### build_opencl.bat

A program OpenCL-es változatát fordítja és futtatja.

## Videó készítése

A szimuláció képkockái a "frames" mappába kerülnek ".ppm" formátumban.

A képkockákból először létrehozható a "frames.txt" fájl:

"
Get-ChildItem frames\*.ppm | Sort-Object Name | ForEach-Object { "file '$($_.FullName)'" } > frames.txt
"

Ezután az FFmpeg segítségével MP4 videó készíthető:

"
ffmpeg -framerate 30 -start_number 1 -i frames/frame_%05d.ppm -pix_fmt yuv420p fluid.mp4
"

## A két módszer összehasonlítása

A CPU-s és OpenCL-es módszer összehasonlítása a következő fájlokban tekinthető meg:

* "FluidSimEvaluation.pdf"
* "FluidSimEvaluation.xlsx"