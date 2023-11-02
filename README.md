# OCR_craft

Project based on this blog - [How to Extract Texts from Rotated(Skewed) Text-Images using CRAFT, OpenCV and pytesseract](https://ai.plainenglish.io/how-to-extract-texts-from-rotated-skewed-text-images-using-craft-opencv-and-pytesseract-9c8c3fb8ef9d) and [github]( https://github.com/clovaai/CRAFT-pytorch )

Video demo [here](https://drive.google.com/file/d/1l5_N1BgAj9N5rP1xvuhJ5W0w7WOpTyo_/view?usp=sharing)

## Overview

Detects and corrects skew and performs OCR.  This version updates the CRAFT project with updates to requirements.txt and changes to account for deprecated code

## Details

Program reads from a directory specified by the user on the command line and outputs deskewed images to the results file.  In addition, detected text will be printed.

### Weights

Weights are staged in the ./weights directory and are sent in via the --trained_model parameter.

### Cuda

Cuda is TRUE by default, so set to False if you're on a laptop

## Execution

python test_craft.py --trained_model=<"path to weights file"> --test_folder=<"path to input images"> --cuda="False | True"

## Output

==================================

Processing image: <image path>
Angle:  < angle of skew detected >
Image to text: 

OCR text from Pytesseract


==================================
