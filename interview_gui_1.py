from os import lseek
import tkinter as tk
from tkinter import filedialog
from tkinter.constants import N
from PIL import Image, ImageTk
import sounddevice as sd
import numpy as np
import wavio
import os

print(sd.query_devices())

os.chdir("/Volumes/Ryan_Extern/Normal_Dataset_Full/Training_Data/bound_images/")

class ImageDescriptionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Interaction with Common Household Items")
        self.geometry("600x700")

        self.image_path = None
        self.image_index = 0
        self.image_list = []
        self.description_list = []

        # All question options:
        # self.instruction_labels = ['Provide the following instruction: Close the window', 'Provide the following instruction: Open the window', 'Provide the following instruction: Go to the window', 'Provide the following instruction: Close the door', 'Provide the following instruction: Open the door', 'Provide the following instruction: Go to the door', 'Provide the following instruction: Where is the light', 'Provide the following instruction: Turn on the light', 'Provide the following instruction: Clean the carpet', 'Provide the following instruction: Close the cupboard', 'Provide the following instruction: Open the cupboard', 'Provide the following instruction: Where is the cupboard', 'Provide the following instruction: Go to the cupboard', 'Provide the following instruction: Where is the mirror', 'Provide the following instruction: Go to the mirror', 'Provide the following instruction: Close the curtains.', 'Provide the following instruction: Clean the curtains.', 'Provide the following instruction: Open the curtains.', 'Provide the following instruction: Close the Dustbin', 'Provide the following instruction: Open the Dustbin', 'Provide the following instruction: Where is the Dustbin', 'Provide the following instruction: Go to the Dustbin', 'Provide the following instruction: Close the Fridge', 'Provide the following instruction: Open the Fridge', 'Provide the following instruction: Where is the Fridge', 'Provide the following instruction: Go to the Fridge', 'Provide the following instruction: Close the Oven', 'Provide the following instruction: Open the Oven', 'Provide the following instruction: Where is the Oven', 'Provide the following instruction: Go to the Oven', 'Provide the following instruction: Turn on the Oven', 'Provide the following instruction: Close the Microwave', 'Provide the following instruction: Open the Microwave', 'Provide the following instruction: Where is the Microwave', 'Provide the following instruction: Turn on the Microwave', 'Provide the following instruction: Fetch the Kettle', 'Provide the following instruction: Pick up the Kettle', 'Provide the following instruction: Where is the Kettle', 'Provide the following instruction: Go to the Kettle', 'Provide the following instruction: Turn on the Kettle', 'Provide the following instruction: Where is the Toaster', 'Provide the following instruction: Go to the Toaster', 'Provide the following instruction: Turn on the Toaster', 'Provide the following instruction: Close the Tap', 'Provide the following instruction: Open the Tap', 'Provide the following instruction: Where is the Tap', 'Provide the following instruction: Turn on the Tap', 'Provide the following instruction: Where is the Sink', 'Provide the following instruction: Go to the Sink', 'Provide the following instruction: Fetch the Pot', 'Provide the following instruction: Pick up the Pot', 'Provide the following instruction: Where is the Pot', 'Provide the following instruction: Fetch the Cup', 'Provide the following instruction: Pick up the Cup', 'Provide the following instruction: Throw away the Cup', 'Provide the following instruction: Where is the Cup', 'Provide the following instruction: Fetch the Bowl', 'Provide the following instruction: Pick up the Bowl', 'Provide the following instruction: Throw away the Bowl', 'Provide the following instruction: Where is the Bowl', 'Provide the following instruction: Where is the Toilet', 'Provide the following instruction: Close the Dishwasher', 'Provide the following instruction: Open the Dishwasher', 'Provide the following instruction: Where is the Dishwasher', 'Provide the following instruction: Go to the Dishwasher', 'Provide the following instruction: Turn on the Dishwasher', 'Provide the following instruction: Fetch the Cloth', 'Provide the following instruction: Pick up the Cloth', 'Provide the following instruction: Throw away the Cloth', 'Provide the following instruction: Where is the Cloth', 'Provide the following instruction: Fetch the Jug', 'Provide the following instruction: Pick up the Jug', 'Provide the following instruction: Where is the Jug', 'Provide the following instruction: Fetch the Bottle', 'Provide the following instruction: Pick up the Bottle', 'Provide the following instruction: Throw away the Bottle', 'Provide the following instruction: Where is the Bottle', 'Provide the following instruction: Where is the CoffeeMachine', 'Provide the following instruction: Go to the CoffeeMachine', 'Provide the following instruction: Turn on the CoffeeMachine', 'Provide the following instruction: Go to the Couch', 'Provide the following instruction: Where is the Television', 'Provide the following instruction: Go to the Television', 'Provide the following instruction: Turn on the Television', 'Provide the following instruction: Fetch the Pillow', 'Provide the following instruction: Pick up the Pillow', 'Provide the following instruction: Where is the Pillow', 'Provide the following instruction: Go to the FirePlace', 'Provide the following instruction: Go to the Table', 'Provide the following instruction: Where is the Bed', 'Provide the following instruction: Go to the Bed', 'Provide the following instruction: Close the Draws', 'Provide the following instruction: Open the Draws', 'Provide the following instruction: Where is the Desk', 'Provide the following instruction: Go to the Desk', 'Provide the following instruction: Fetch the Computer', 'Provide the following instruction: Pick up the Computer', 'Provide the following instruction: Where is the Computer', 'Provide the following instruction: Go to the Computer', 'Provide the following instruction: Turn on the Computer', 'Provide the following instruction: Fetch the Flowers', 'Provide the following instruction: Pick up the Flowers', 'Provide the following instruction: Throw away the Flowers', 'Provide the following instruction: Clean the Flowers', 'Provide the following instruction: Where is the Flowers']
        
        # 50 Questions:
        #self.instruction_labels = ['Provide the following instruction: Go to the Window', 'Provide the following instruction: Where is the Toilet', 'Provide the following instruction: Go to the Fridge', 'Provide the following instruction: Go to the Dustbin', 'Provide the following instruction: Open the Oven', 'Provide the following instruction: Where is the Mirror', 'Provide the following instruction: Close the Dishwasher', 'Provide the following instruction: Fetch the Computer', 'Provide the following instruction: Where is the Cup', 'Provide the following instruction: Close the Dustbin', 'Provide the following instruction: Open the Fridge', 'Provide the following instruction: Go to the Couch', 'Provide the following instruction: Where is the Bed', 'Provide the following instruction: Where is the Kettle', 'Provide the following instruction: Where is the Fridge', 'Provide the following instruction: Where is the Cupboard', 'Provide the following instruction: Pick up the Cloth', 'Provide the following instruction: Where is the Jug', 'Provide the following instruction: Open the Dishwasher', 'Provide the following instruction: Close the Microwave', 'Provide the following instruction: Go to the Sink', 'Provide the following instruction: Where is the Television', 'Provide the following instruction: Where is the Bottle', 'Provide the following instruction: Where is the Pot', 'Provide the following instruction: Fetch the Pot', 'Provide the following instruction: Open the Draws', 'Provide the following instruction: Go to the FirePlace', 'Provide the following instruction: Open the Window', 'Provide the following instruction: Open the Curtains', 'Provide the following instruction: Pick up the Bottle', 'Provide the following instruction: Where is the Oven', 'Provide the following instruction: Go to the Toaster', 'Provide the following instruction: Where is the Dishwasher', 'Provide the following instruction: Where is the Light', 'Provide the following instruction: Fetch the Kettle', 'Provide the following instruction: Open the Cupboard', 'Provide the following instruction: Fetch the Bowl', 'Provide the following instruction: Pick up the Jug', 'Provide the following instruction: Go to the Oven', 'Provide the following instruction: Where is the Microwave', 'Provide the following instruction: Where is the CoffeeMachine', 'Provide the following instruction: Go to the Dishwasher', 'Provide the following instruction: Go to the Television', 'Provide the following instruction: Where is the Sink', 'Provide the following instruction: Close the Door', 'Provide the following instruction: Pick up the Kettle', 'Provide the following instruction: Go to the Computer', 'Provide the following instruction: Pick up the Computer', 'Provide the following instruction: Fetch the Cup', 'Provide the following instruction: Where is the Dustbin']

        # 100 Questions Modified:
        self.instruction_labels = ['Provide the following instruction: Go to the Window', 'Provide the following instruction: Where is the Toilet', 'Provide the following instruction: Go to the Fridge', 'Provide the following instruction: Go to the Dustbin', 'Provide the following instruction: Open the Oven', 'Provide the following instruction: Where is the Mirror', 'Provide the following instruction: Close the Dishwasher', 'Provide the following instruction: Fetch the Computer', 'Provide the following instruction: Throw away the Cup', 'Provide the following instruction: Close the Dustbin', 'Provide the following instruction: Open the Fridge', 'Provide the following instruction: Go to the Couch', 'Provide the following instruction: Where is the Bed', 'Provide the following instruction: Turn on the Tap', 'Provide the following instruction: Fetch the Pillow', 'Provide the following instruction: Where is the Cupboard', 'Provide the following instruction: Pick up the Cloth', 'Provide the following instruction: Where is the Jug', 'Provide the following instruction: Open the Dishwasher', 'Provide the following instruction: Close the Microwave', 'Provide the following instruction: Go to the Sink', 'Provide the following instruction: Where is the Television', 'Provide the following instruction: Where is the Bottle', 'Provide the following instruction: Throw away the Pot', 'Provide the following instruction: Fetch the Pot', 'Provide the following instruction: Open the Draws', 'Provide the following instruction: Go to the FirePlace', 'Provide the following instruction: Open the Window', 'Provide the following instruction: Open the Curtains', 'Provide the following instruction: Pick up the Bottle', 'Provide the following instruction: Turn on the Microwave', 'Provide the following instruction: Go to the Toaster', 'Provide the following instruction: Go to the Table', 'Provide the following instruction: Where is the Light', 'Provide the following instruction: Fetch the Kettle', 'Provide the following instruction: Open the Cupboard', 'Provide the following instruction: Fetch the Bowl', 'Provide the following instruction: Pick up the Jug', 'Provide the following instruction: Turn on the Oven', 'Provide the following instruction: Where is the Microwave', 'Provide the following instruction: Where is the CoffeeMachine', 'Provide the following instruction: Go to the Desk', 'Provide the following instruction: Go to the Television', 'Provide the following instruction: Where is the Sink', 'Provide the following instruction: Close the Door', 'Provide the following instruction: Pick up the Kettle', 'Provide the following instruction: Clean the Carpet', 'Provide the following instruction: Pick up the Computer', 'Provide the following instruction: Fetch the Cup', 'Provide the following instruction: Throw away the Flowers', 'Provide the following instruction: Go to the Window', 'Provide the following instruction: Where is the Toilet', 'Provide the following instruction: Go to the Fridge', 'Provide the following instruction: Go to the Dustbin', 'Provide the following instruction: Open the Oven', 'Provide the following instruction: Where is the Mirror', 'Provide the following instruction: Close the Dishwasher', 'Provide the following instruction: Fetch the Computer', 'Provide the following instruction: Throw away the Cup', 'Provide the following instruction: Close the Dustbin', 'Provide the following instruction: Open the Fridge', 'Provide the following instruction: Go to the Couch', 'Provide the following instruction: Where is the Bed', 'Provide the following instruction: Turn on the Tap', 'Provide the following instruction: Fetch the Pillow', 'Provide the following instruction: Where is the Cupboard', 'Provide the following instruction: Pick up the Cloth', 'Provide the following instruction: Where is the Jug', 'Provide the following instruction: Open the Dishwasher', 'Provide the following instruction: Close the Microwave', 'Provide the following instruction: Go to the Sink', 'Provide the following instruction: Where is the Television', 'Provide the following instruction: Where is the Bottle', 'Provide the following instruction: Throw away the Pot', 'Provide the following instruction: Fetch the Pot', 'Provide the following instruction: Open the Draws', 'Provide the following instruction: Go to the FirePlace', 'Provide the following instruction: Open the Window', 'Provide the following instruction: Open the Curtains', 'Provide the following instruction: Pick up the Bottle', 'Provide the following instruction: Turn on the Microwave', 'Provide the following instruction: Go to the Toaster', 'Provide the following instruction: Go to the Table', 'Provide the following instruction: Where is the Light', 'Provide the following instruction: Fetch the Kettle', 'Provide the following instruction: Open the Cupboard', 'Provide the following instruction: Fetch the Bowl', 'Provide the following instruction: Pick up the Jug', 'Provide the following instruction: Turn on the Oven', 'Provide the following instruction: Where is the Microwave', 'Provide the following instruction: Where is the CoffeeMachine', 'Provide the following instruction: Go to the Desk', 'Provide the following instruction: Go to the Television', 'Provide the following instruction: Where is the Sink', 'Provide the following instruction: Close the Door', 'Provide the following instruction: Pick up the Kettle', 'Provide the following instruction: Clean the Carpet', 'Provide the following instruction: Pick up the Computer', 'Provide the following instruction: Fetch the Cup', 'Provide the following instruction: Throw away the Flowers', 'Provide the following instruction: Go to the Window', 'Provide the following instruction: Where is the Toilet', 'Provide the following instruction: Go to the Fridge', 'Provide the following instruction: Go to the Dustbin', 'Provide the following instruction: Open the Oven', 'Provide the following instruction: Where is the Mirror', 'Provide the following instruction: Close the Dishwasher', 'Provide the following instruction: Fetch the Computer', 'Provide the following instruction: Throw away the Cup', 'Provide the following instruction: Close the Dustbin', 'Provide the following instruction: Open the Fridge', 'Provide the following instruction: Go to the Couch', 'Provide the following instruction: Where is the Bed', 'Provide the following instruction: Turn on the Tap', 'Provide the following instruction: Fetch the Pillow', 'Provide the following instruction: Where is the Cupboard', 'Provide the following instruction: Pick up the Cloth', 'Provide the following instruction: Where is the Jug', 'Provide the following instruction: Open the Dishwasher', 'Provide the following instruction: Close the Microwave', 'Provide the following instruction: Go to the Sink', 'Provide the following instruction: Where is the Television', 'Provide the following instruction: Where is the Bottle', 'Provide the following instruction: Throw away the Pot', 'Provide the following instruction: Fetch the Pot', 'Provide the following instruction: Open the Draws', 'Provide the following instruction: Go to the FirePlace', 'Provide the following instruction: Open the Window', 'Provide the following instruction: Open the Curtains', 'Provide the following instruction: Pick up the Bottle', 'Provide the following instruction: Turn on the Microwave', 'Provide the following instruction: Go to the Toaster', 'Provide the following instruction: Go to the Table', 'Provide the following instruction: Where is the Light', 'Provide the following instruction: Fetch the Kettle', 'Provide the following instruction: Open the Cupboard', 'Provide the following instruction: Fetch the Bowl', 'Provide the following instruction: Pick up the Jug', 'Provide the following instruction: Turn on the Oven', 'Provide the following instruction: Where is the Microwave', 'Provide the following instruction: Where is the CoffeeMachine', 'Provide the following instruction: Go to the Desk', 'Provide the following instruction: Go to the Television', 'Provide the following instruction: Where is the Sink', 'Provide the following instruction: Close the Door', 'Provide the following instruction: Pick up the Kettle', 'Provide the following instruction: Clean the Carpet', 'Provide the following instruction: Pick up the Computer', 'Provide the following instruction: Fetch the Cup', 'Provide the following instruction: Throw away the Flowers']

        self.start_screen()

    def start_screen(self):
        self.start_frame = tk.Frame(self)
        self.start_frame.pack(pady=20)

        self.start_label = tk.Label(self.start_frame, text="Welcome!\n This application serves the purpose of allowing you, the interviewee, \n to give instructions for an individual to complete common household tasks based \n on the image and highlighted object you are shown.\n This process works as follows: \n\n 1. You will be presented with an image and a task to complete. \n 2. To enter a text instruction press on the textbox and type your instruction.\n 3. To add a voice recording press 'Record' and speak.\n 4. To go to the next question click on the 'Next' button. \n 5. Upon completion of all questions you can press 'Quit' to exit the application.", font=("Helvetica", 16))
        self.start_label.pack(pady=10)

        self.start_button = tk.Button(self.start_frame, text="Start", command=self.show_next_image)
        self.start_button.pack(pady=5)

    def load_images(self):
        # All quesion options:
        # self.image_list = ['IMG_0004.jpeg', 'IMG_0005.jpeg', 'IMG_0006.jpeg', 'IMG_0002.jpeg', 'IMG_0013.jpeg', 'IMG_0018.jpeg', 'IMG_0039.jpeg', 'IMG_0040.jpeg', 'IMG_0037.jpeg', 'IMG_0008.jpeg', 'IMG_0016.jpeg', 'IMG_0017.jpeg', 'IMG_0030.jpeg', 'IMG_0014.jpeg', 'IMG_0015.jpeg', 'IMG_0019.jpeg', 'IMG_0038.jpeg', 'IMG_0046.jpeg', 'IMG_0029.jpeg', 'IMG_0031.jpeg', 'IMG_0032.jpeg', 'IMG_0033.jpeg', 'IMG_0354.jpeg', 'IMG_0369.jpeg', 'IMG_0370.jpeg', 'IMG_0371.jpeg', 'IMG_0043.jpeg', 'IMG_0045.jpeg', 'IMG_0358.jpeg', 'IMG_0359.jpeg', 'IMG_0364.jpeg', 'IMG_0034.jpeg', 'IMG_0360.jpeg', 'IMG_0361.jpeg', 'IMG_0362.jpeg', 'IMG_0365.jpeg', 'IMG_0374.jpeg', 'IMG_0375.jpeg', 'IMG_0376.jpeg', 'IMG_0377.jpeg', 'IMG_0379.jpeg', 'IMG_0380.jpeg', 'IMG_0381.jpeg', 'IMG_0021.jpeg', 'IMG_0022.jpeg', 'IMG_0028.jpeg', 'IMG_0198.jpeg', 'IMG_0023.jpeg', 'IMG_0199.jpeg', 'IMG_5049.jpeg', 'IMG_5050.jpeg', 'IMG_5051.jpeg', 'IMG_0280.jpeg', 'IMG_0291.jpeg', 'IMG_0391.jpeg', 'IMG_4947.jpeg', 'IMG_0363.jpeg', 'IMG_0373.jpeg', 'IMG_5052.jpeg', 'IMG_5053.jpeg', 'IMG_0024.jpeg', 'IMG_0405.jpeg', 'IMG_0406.jpeg', 'IMG_0407.jpeg', 'IMG_0414.jpeg', 'IMG_0415.jpeg', 'IMG_0200.jpeg', 'IMG_0201.jpeg', 'IMG_0202.jpeg', 'IMG_0217.jpeg', 'IMG_5029.jpeg', 'IMG_5030.jpeg', 'IMG_5031.jpeg', 'IMG_0216.jpeg', 'IMG_0290.jpeg', 'IMG_0309.jpeg', 'IMG_0388.jpeg', 'IMG_0355.jpeg', 'IMG_0356.jpeg', 'IMG_0384.jpeg', 'IMG_0041.jpeg', 'IMG_0140.jpeg', 'IMG_0141.jpeg', 'IMG_0146.jpeg', 'IMG_0048.jpeg', 'IMG_0049.jpeg', 'IMG_0050.jpeg', 'IMG_0443.jpeg', 'IMG_0047.jpeg', 'IMG_0168.jpeg', 'IMG_0169.jpeg', 'IMG_0035.jpeg', 'IMG_0036.jpeg', 'IMG_5017.jpeg', 'IMG_5018.jpeg', 'IMG_0324.jpeg', 'IMG_0325.jpeg', 'IMG_4928.jpeg', 'IMG_4933.jpeg', 'IMG_5021.jpeg', 'IMG_0051.jpeg', 'IMG_0052.jpeg', 'IMG_0069.jpeg', 'IMG_0070.jpeg', 'IMG_0071.jpeg']
        
        # 50 Questions:
        # self.image_list = ['IMG_0004.jpeg', 'IMG_0005.jpeg', 'IMG_0354.jpeg', 'IMG_0029.jpeg', 'IMG_0043.jpeg', 'IMG_0014.jpeg', 'IMG_0405.jpeg', 'IMG_0324.jpeg', 'IMG_0280.jpeg', 'IMG_0030.jpeg', 'IMG_0369.jpeg', 'IMG_0037.jpeg', 'IMG_0168.jpeg', 'IMG_0364.jpeg', 'IMG_0370.jpeg', 'IMG_0008.jpeg', 'IMG_0200.jpeg', 'IMG_5029.jpeg', 'IMG_0406.jpeg', 'IMG_0034.jpeg', 'IMG_0015.jpeg', 'IMG_0140.jpeg', 'IMG_0021.jpeg', 'IMG_5049.jpeg', 'IMG_5050.jpeg', 'IMG_0035.jpeg', 'IMG_0443.jpeg', 'IMG_0006.jpeg', 'IMG_0016.jpeg', 'IMG_0022.jpeg', 'IMG_0045.jpeg', 'IMG_0379.jpeg', 'IMG_0407.jpeg', 'IMG_0039.jpeg', 'IMG_0365.jpeg', 'IMG_0017.jpeg', 'IMG_0363.jpeg', 'IMG_5030.jpeg', 'IMG_0358.jpeg', 'IMG_0360.jpeg', 'IMG_0355.jpeg', 'IMG_0414.jpeg', 'IMG_0141.jpeg', 'IMG_0018.jpeg', 'IMG_0002.jpeg', 'IMG_0374.jpeg', 'IMG_0325.jpeg', 'IMG_4928.jpeg', 'IMG_0291.jpeg', 'IMG_0031.jpeg']

        # 100 Questions Modified:
        self.image_list = ['IMG_0004.jpeg', 'IMG_0005.jpeg', 'IMG_0354.jpeg', 'IMG_0029.jpeg', 'IMG_0043.jpeg', 'IMG_0014.jpeg', 'IMG_0405.jpeg', 'IMG_0324.jpeg', 'IMG_0280.jpeg', 'IMG_0030.jpeg', 'IMG_0369.jpeg', 'IMG_0037.jpeg', 'IMG_0168.jpeg', 'IMG_0016.jpeg', 'IMG_0048.jpeg', 'IMG_0008.jpeg', 'IMG_0200.jpeg', 'IMG_5029.jpeg', 'IMG_0406.jpeg', 'IMG_0034.jpeg', 'IMG_0015.jpeg', 'IMG_0140.jpeg', 'IMG_0021.jpeg', 'IMG_5049.jpeg', 'IMG_5050.jpeg', 'IMG_0035.jpeg', 'IMG_0443.jpeg', 'IMG_0006.jpeg', 'IMG_0017.jpeg', 'IMG_0022.jpeg', 'IMG_0360.jpeg', 'IMG_0379.jpeg', 'IMG_0002.jpeg', 'IMG_0039.jpeg', 'IMG_0364.jpeg', 'IMG_0018.jpeg', 'IMG_0363.jpeg', 'IMG_5030.jpeg', 'IMG_0045.jpeg', 'IMG_0361.jpeg', 'IMG_0355.jpeg', 'IMG_5017.jpeg', 'IMG_0141.jpeg', 'IMG_0023.jpeg', 'IMG_0013.jpeg', 'IMG_0365.jpeg', 'IMG_0038.jpeg', 'IMG_0325.jpeg', 'IMG_0291.jpeg', 'IMG_0032.jpeg', 'IMG_0011.jpeg', 'IMG_0019.jpeg', 'IMG_0370.jpeg', 'IMG_0031.jpeg', 'IMG_0358.jpeg', 'IMG_0026.jpeg', 'IMG_0407.jpeg', 'IMG_4928.jpeg', 'IMG_0391.jpeg', 'IMG_0033.jpeg', 'IMG_0371.jpeg', 'IMG_0040.jpeg', 'IMG_0169.jpeg', 'IMG_0028.jpeg', 'IMG_0049.jpeg', 'IMG_0059.jpeg', 'IMG_0201.jpeg', 'IMG_5031.jpeg', 'IMG_0414.jpeg', 'IMG_0362.jpeg', 'IMG_0199.jpeg', 'IMG_0146.jpeg', 'IMG_0216.jpeg', 'IMG_5051.jpeg', 'IMG_5052.jpeg', 'IMG_0036.jpeg', 'IMG_0444.jpeg', 'IMG_0024.jpeg', 'IMG_0046.jpeg', 'IMG_0217.jpeg', 'IMG_0378.jpeg', 'IMG_0380.jpeg', 'IMG_0041.jpeg', 'IMG_0065.jpeg', 'IMG_0374.jpeg', 'IMG_0060.jpeg', 'IMG_0373.jpeg', 'IMG_5032.jpeg', 'IMG_0359.jpeg', 'IMG_0396.jpeg', 'IMG_0356.jpeg', 'IMG_5018.jpeg', 'IMG_0147.jpeg', 'IMG_0206.jpeg', 'IMG_0020.jpeg', 'IMG_0375.jpeg', 'IMG_0047.jpeg', 'IMG_4933.jpeg', 'IMG_4947.jpeg', 'IMG_0051.jpeg', 'IMG_0004.jpeg', 'IMG_0005.jpeg', 'IMG_0354.jpeg', 'IMG_0029.jpeg', 'IMG_0043.jpeg', 'IMG_0014.jpeg', 'IMG_0405.jpeg', 'IMG_0324.jpeg', 'IMG_0280.jpeg', 'IMG_0030.jpeg', 'IMG_0369.jpeg', 'IMG_0037.jpeg', 'IMG_0168.jpeg', 'IMG_0016.jpeg', 'IMG_0048.jpeg', 'IMG_0008.jpeg', 'IMG_0200.jpeg', 'IMG_5029.jpeg', 'IMG_0406.jpeg', 'IMG_0034.jpeg', 'IMG_0015.jpeg', 'IMG_0140.jpeg', 'IMG_0021.jpeg', 'IMG_5049.jpeg', 'IMG_5050.jpeg', 'IMG_0035.jpeg', 'IMG_0443.jpeg', 'IMG_0006.jpeg', 'IMG_0017.jpeg', 'IMG_0022.jpeg', 'IMG_0360.jpeg', 'IMG_0379.jpeg', 'IMG_0002.jpeg', 'IMG_0039.jpeg', 'IMG_0364.jpeg', 'IMG_0018.jpeg', 'IMG_0363.jpeg', 'IMG_5030.jpeg', 'IMG_0045.jpeg', 'IMG_0361.jpeg', 'IMG_0355.jpeg', 'IMG_5017.jpeg', 'IMG_0141.jpeg', 'IMG_0023.jpeg', 'IMG_0013.jpeg', 'IMG_0365.jpeg', 'IMG_0038.jpeg', 'IMG_0325.jpeg', 'IMG_0291.jpeg', 'IMG_0032.jpeg']
    
    def create_widgets(self):
        self.image_label = tk.Label(self)
        self.image_label.pack(pady=10)

        self.instruct_label = tk.Label(self, text=self.instruction_labels[self.image_index], font=("Helvetica", 24))
        self.instruct_label.pack(pady=5)

        self.description_entry = tk.Entry(self, width=40)
        self.description_entry.pack(pady=5)

        self.record_button = tk.Button(self, text="Record", command=self.record_audio)
        self.record_button.pack(side=tk.LEFT, padx=5, pady=0)

        self.save_button = tk.Button(self, text="Next", command=self.save_description)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=0)

        self.quit_button = tk.Button(self, text="Quit", command=self.quit)
        self.quit_button.pack(side=tk.RIGHT, padx=5, pady=0)

        self.update_image()

    def record_audio(self):
        fs = 16000  # Sample rate
        duration = 3.3  # Recording duration 
        input_device = 0 # Input Channel

        # Record audio using sounddevice
        audio_data = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype=np.int16, device=input_device)
        sd.wait()

        # Save the audio as a WAV file
        filename = f"recordings/34101_{str(self.image_index).zfill(3)}.wav"
        wavio.write(filename, audio_data, fs, sampwidth=2)
    
    def show_next_image(self):
        if self.image_index == 0:
            self.start_frame.pack_forget()
            self.load_images()
            self.create_widgets()
        elif self.image_index == len(self.image_list):
            self.end_screen()
            return
        else:
            self.update_image()

    def update_image(self):
        if self.image_index < len(self.image_list):
            image_path = self.image_list[self.image_index]
            image = Image.open(image_path)
            image = image.resize((500, 500))
            #image = image.rotate(270)
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo
            self.instruct_label.config(text=self.instruction_labels[self.image_index])

        else:
            self.image_label.config(image=None)
            self.description_entry.delete(0, tk.END)
            self.save_button.config(state=tk.DISABLED)

    def save_description(self):
        description = self.description_entry.get()
        if description:
            self.description_list.append(description)
            self.description_entry.delete(0, tk.END)
            self.image_index += 1
            self.show_next_image()
        self.save_to_textfile()

    def end_screen(self):
        self.end_frame = tk.Frame(self)
        self.end_frame.pack(pady=20)

        self.end_label = tk.Label(self.end_frame, text="Thank you for your participation.", font=("Helvetica", 16))
        self.end_label.pack(pady=10)

    def save_to_textfile(self):
        with open("image_descriptions.txt", "w") as file:
            for description in self.description_list:
                file.write(description + "\n")

if __name__ == "__main__":
    app = ImageDescriptionApp()
    app.mainloop()