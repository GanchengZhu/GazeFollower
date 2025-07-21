# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import tkinter as tk
from tkinter import ttk


class ParticipantInfoDialog:
    def __init__(self, root_):
        self.root = root_
        self.root.title("Participant Information")

        # Variables to store the inputs
        self.participant_id = tk.StringVar(value="ABC")
        self.age = tk.IntVar(value=25)
        self.gender = tk.StringVar(value="M")
        self.glasses = tk.IntVar(value=1)

        # Create the input fields
        self.create_widgets()

        # Center the window
        self.center_window(540, 340)  # Adjusted width and height for the new layout

    def create_widgets(self):
        # ID Label and Entry
        label_id = ttk.Label(self.root, text="Participant ID:", font=("Microsoft YaHei", 14))
        label_id.grid(row=0, column=0, padx=20, pady=15, sticky="w")
        entry_id = ttk.Entry(self.root, textvariable=self.participant_id, font=("Microsoft YaHei", 14))
        entry_id.grid(row=0, column=1, columnspan=2, padx=20, pady=15, sticky="ew")

        # Age Label and Entry
        label_age = ttk.Label(self.root, text="Age:", font=("Microsoft YaHei", 14))
        label_age.grid(row=1, column=0, padx=20, pady=15, sticky="w")
        entry_age = ttk.Entry(self.root, textvariable=self.age, font=("Microsoft YaHei", 14))
        entry_age.grid(row=1, column=1, columnspan=2, padx=20, pady=15, sticky="ew")

        # Gender Label and Combobox
        label_gender = ttk.Label(self.root, text="Gender:", font=("Microsoft YaHei", 14))
        label_gender.grid(row=2, column=0, padx=20, pady=15, sticky="w")
        combo_gender = ttk.Combobox(self.root, textvariable=self.gender, state="readonly", font=("Microsoft YaHei", 14))
        combo_gender['values'] = ("M", "F", "O")
        combo_gender.grid(row=2, column=1, columnspan=2, padx=20, pady=15, sticky="ew")
        combo_gender.current(0)  # Default to "Male"

        # Glasses Label and ComboboxZ
        label_glasses = ttk.Label(self.root, text="Wears Glasses:", font=("Microsoft YaHei", 14))
        label_glasses.grid(row=3, column=0, padx=20, pady=15, sticky="w")
        combo_glasses = ttk.Combobox(self.root, textvariable=self.glasses,
                                     state="readonly", font=("Microsoft YaHei", 14))
        combo_glasses['values'] = ("0", "1")
        combo_glasses.grid(row=3, column=1, columnspan=2, padx=20, pady=15, sticky="ew")
        combo_glasses.current(1)  # Default to "No"

        # Submit button
        submit_button = tk.Button(self.root, text="Enter", command=self.submit, font=("Microsoft YaHei", 14), width=20)
        submit_button.grid(row=4, column=0, columnspan=3, padx=20, pady=20)

        # Adjust column weights to ensure proper resizing
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)

    def center_window(self, width, height):
        # Get the screen width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Calculate the position for the window to be centered
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)

        # Set the window size and position
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def submit(self):
        # Close the window and return the input values
        self.root.destroy()

    def get_info(self):
        # Return the entered participant information as a dictionary
        return {
            "subject_id": self.participant_id.get(),
            "age": self.age.get(),
            "gender": self.gender.get(),
            "wears_glasses": self.glasses.get()
        }


# To use this class:
if __name__ == "__main__":
    root = tk.Tk()
    dialog = ParticipantInfoDialog(root)
    root.mainloop()  # Start the Tkinter main loop

    # Get the results after the dialog is closed
    participant_info = dialog.get_info()
    print(participant_info)
