# video-labeler
Necessary tool for the data preparation of the soccer highlights project (https://github.com/gioele-maruccia/deep-learning-soccer-highlights).

## Functionalities

Given a whole match video, this tool allows to: 
1) select the relevant events.
2) attach to them the match time using an OCR.
3) cut the selected events.

## Install

Install the necessary libraries

`pip install -r requirements.txt`

###For Linux users only:

Please comment this line from the src/text_recognition.py file: 
`pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'`


###For windows users only: 

Install tesseract (from here https://github.com/UB-Mannheim/tesseract/wiki) and make sure the installation folder is C:\Program Files (x86)\Tesseract-OCR\tesseract.exe. 
If that's not the case, please correct the tesseract reference in the first lines of the src/text_recognition.py file: 

`pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'`


## GUI

![demo](demo.png)

## How it works
- Click "File" from the toolbar and select the soccer video to be inported. 
- Using the mouse, draw a rectangle around the time flag (usually located on the top-left corner).
- Select, one by one, the video sections to be cut by selecting their init time, final time and when the relevant event has occurred. Also relevant to select the starting point of the celebration. 
- At this point, once verified the init time and stop time informations, add the section using the add button. Eventually, delete wrong video sections using the delete button. 
- Once all video sections have been selected, you can press the "cut" button to generate your video cuts and to save them in the "cuts" folder.
- Moreover, a "label_info.csv" file will be generated inside the "cuts" folder with this header in order to take into account the relevant video informations:
  - video_name,
  - N_highlight,
  - starting_frame,
  - goal_frame,
  - ending_frame,
  - start_celebration,
  - starting_time,
  - ending_time,
  - added_frames_bf,
  - fps

## Notes
- The optical character recognition is useful as a support for labeling but it not always works as expected. So please double check the video timestamps before saving each video section.
- The input file name should starts with the string "complete" in order to make the cut process works. This can be changed inside the app.py cut_videos function. 
- Each chosen video section should represent a single goal but for the needs of the neural network we cut a broader video section (with a padding of 30 seconds) so to take into account also the non relevant aspects of the soccer video.

## The dataset
With the help of this GUI, I created a dataset made of 271 goals coming from the "La Liga" championship.
You can find them, togheter with the labels, inside [this](https://www.dropbox.com/sh/up35305tbk64p4k/AACZnQF3lECKs_LexvZ8Fej3a?dl=0) dropbox directory, under the "full_game_goal_cuts" folder. 