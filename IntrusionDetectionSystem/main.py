import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import subprocess
import os

def execute_assignment(assignment_file):
    """
    Executes the given assignment file.
    
    Args:
        assignment_file (str): The path to the Python or IPython Notebook file.
    """
    process = subprocess.Popen(['python', assignment_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = process.communicate()
    if errors:
        print("Error:", errors.decode())

def select_assignment_file():
    """
    Opens a file dialog to select a Python or IPython Notebook file
    and executes it.
    """
    assignment_file = filedialog.askopenfilename(filetypes=[("Python files", "*.py"), ("IPython Notebook files", "*.ipynb")])
    if assignment_file:
        execute_assignment(assignment_file)

def setup_gui():
    """
    Sets up the Tkinter GUI for the Intrusion Detection System.
    """
    # GUI setup
    root = tk.Tk()
    root.title("Intrusion Detection System GUI")
    root.geometry("800x600")

    # Information space at the top
    info_label1 = tk.Label(root, text="Video Analytics\n Intrusion Detection System- Prototype\n DILLIP S", font=("Helvetica", 16))
    info_label1.pack(pady=20)

    info_label2 = tk.Label(root, text="Press 'q' to exit output", font=("Helvetica", 8))
    info_label2.pack(pady=20)

    # Create buttons for each assignment
    assignments = {
        "Background Model from Image Sequence": "BackgroundModel.py",
        "Background Model - Custom Video": "BackgroundModelVideo.py",
        "Background Model - Webcam": "BackgroundModelWebcam.py",
        "Scene Change Detection": "sceneDetection.py",
        "Intrusion Detection": "IntrusionDetectionSystem.py",
        "Edge and Corner Detection" : "EdgeCorner.py",
        # "Image Histogram": "A3_q1.py",
        # "Image search based on histogram match": "234102304_A3_Q3.ipynb",
        "Optical Flow": "optical_flow.py",
        # Add more assignments here
    }

    for assignment_name, assignment_file in assignments.items():
        button = tk.Button(root, text=assignment_name, command=lambda file=assignment_file: execute_assignment(file), bg="lightblue", fg="black", font=("Helvetica", 12))
        button.pack(pady=5)

    # Button to select assignment file
    custom_assignment_button = tk.Button(root, text="Custom Scripts", command=select_assignment_file, bg="lightgreen", fg="black", font=("Helvetica", 12))
    custom_assignment_button.pack(pady=10)

    return root

def main():
    """
    Main function to run the Tkinter GUI application.
    """
    root = setup_gui()
    root.mainloop()

if __name__ == "__main__":
    main()
