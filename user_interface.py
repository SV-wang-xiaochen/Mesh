import tkinter as tk
from tkinter import messagebox

INTERACTIVE_INPUT = True  # Set this to False if you want to use default values

def process_input(entries, root, next_window, run_mesh):
    try:
        # Extract values from the entries
        arg1 = entries['arg1'].get()
        arg2 = entries['arg2'].get()

        # Print or use the values
        print("Argument 1:", arg1)
        print("Argument 2:", arg2)
        # Call your function or perform the desired operation with the inputs here

        if run_mesh:
            # Additional code to run after processing the arguments
            print("Running additional code...")

        # Destroy the root window (current window)
        root.destroy()

        # Open the next window for entering another set of parameters
        next_window()

    except ValueError:
        messagebox.showerror("Error", "Please enter valid values for all arguments.")

def open_first_window():
    first_window = tk.Tk()
    first_window.title("First Set of Parameters")

    # Create input fields and get the entries dictionary for the first window
    entries = create_first_window_input_fields(first_window)

    # Create a button that calls the process_input function when clicked
    button_process = tk.Button(first_window, text="Process Input", command=lambda: process_input(entries, first_window, open_second_window, False))
    button_process.grid(row=2, column=0, columnspan=2, pady=10)

def open_second_window():
    second_window = tk.Tk()
    second_window.title("Second Set of Parameters")

    # Create input fields and get the entries dictionary for the second window
    entries = create_second_window_input_fields(second_window)

    # Create a button that calls the process_input function when clicked
    button_process = tk.Button(second_window, text="Process Input", command=lambda: process_input(entries, second_window, open_first_window, True))
    button_process.grid(row=2, column=0, columnspan=2, pady=10)

def create_first_window_input_fields(root):
    # Create a dictionary to store the entry widgets
    entries = {}

    # Create entry widgets for each argument
    arg1_label = tk.Label(root, text="选择模式(1:单组参数 2:遍历参数):")
    arg1_entry = tk.Entry(root)
    entries['arg1'] = arg1_entry

    arg2_label = tk.Label(root, text="选择模式(1:镜片碰撞/遮挡 2:面板碰撞):")
    arg2_entry = tk.Entry(root)
    entries['arg2'] = arg2_entry

    # Position the widgets using the grid layout
    arg1_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
    arg1_entry.grid(row=0, column=1, padx=5, pady=5)

    arg2_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
    arg2_entry.grid(row=1, column=1, padx=5, pady=5)

    return entries

def create_second_window_input_fields(root):
    # Create a dictionary to store the entry widgets
    entries = {}

    # Create entry widgets for each argument
    arg1_label = tk.Label(root, text='面板模型文件名(无后缀):')
    arg1_entry = tk.Entry(root)
    entries['arg1'] = arg1_entry

    arg2_label = tk.Label(root, text='工作距离(周边)[0,45]mm:')
    arg2_entry = tk.Entry(root)
    entries['arg2'] = arg2_entry

    # Position the widgets using the grid layout
    arg1_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
    arg1_entry.grid(row=0, column=1, padx=5, pady=5)

    arg2_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
    arg2_entry.grid(row=1, column=1, padx=5, pady=5)

    return entries

# Open the first window
open_first_window()

# Start the Tkinter event loop
tk.mainloop()

