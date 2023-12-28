import tkinter as tk
from tkinter import messagebox

INTERACTIVE_INPUT = True  # Set this to False if you want to use default values

def process_input(entries, root, next_window, run_mesh):
    try:
        # Extract values from the entries
        args = [entry.get() for entry in entries.values()]

        # Print or use the values
        for i, arg in enumerate(args, start=1):
            print(f"Argument {i}: {arg}")

        # Call your function or perform the desired operation with the inputs here

        if run_mesh:
            # Additional code to run after processing the arguments
            print(f"工作模式：")
            for i, arg in enumerate(args, start=1):
                print(f"Argument {i}: {arg}")
            print("Running additional code...")

        # Destroy the root window (current window)
        root.destroy()

        # Open the next window for entering another set of parameters
        next_window(*args)

    except ValueError:
        messagebox.showerror("Error", "Please enter valid values for all arguments.")

def open_first_window(*args):
    first_window = tk.Tk()
    first_window.title("选择工作模式")

    # Create input fields and get the entries dictionary for the first window
    entries = create_first_window_input_fields(first_window)

    # Create a button that calls the process_input function when clicked
    button_process = tk.Button(first_window, text="Process Input", command=lambda: process_input(entries, first_window, open_second_window, False))
    button_process.grid(row=2, column=0, columnspan=2, pady=10)

def open_second_window(*args):
    second_window = tk.Tk()
    second_window.title("选择工作参数")

    # Create input fields and get the entries dictionary for the second window
    entries = create_second_window_input_fields(second_window, *args)

    # Create a button that calls the process_input function when clicked
    button_process = tk.Button(second_window, text="Process Input", command=lambda: process_input(entries, second_window, lambda *args: open_first_window(*args), True))
    button_process.grid(row=15, column=0, columnspan=2, pady=10)

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

def create_second_window_input_fields(root, *args):
    # Create a dictionary to store the entry widgets
    entries = {}

    arg_label_text = ['工作距离(周边)[0,45]mm:']

    # Use arg1 to determine the number and labels for entry widgets
    if args[1] == '2':
        arg_label_text.append('面板模型文件名(无后缀):')
    if args[1] == '1':
        arg_label_text.append('目镜外框直径[20,80]mm:')
        arg_label_text.append('通光孔径[20,80]mm:')
        arg_label_text.append('单张范围（眼外角）[70,140]度:')

    if args[0] == '1':
        arg_label_text.append('机器俯仰角[-90,90]度(+仰,-俯):')
        arg_label_text.append('机器内外旋角[-90,90]度(+内旋,-外旋):')
        arg_label_text.append('眼睛俯仰角[-90,90]度(+俯,-仰):')
        arg_label_text.append('眼睛内外旋角[-90,90]度(+外旋,-内旋):')
    if args[0] == '2':
        arg_label_text.append('机器俯仰角 最小值(+仰,-俯):')
        arg_label_text.append('机器俯仰角 区间长度:')
        arg_label_text.append('机器俯仰角 步进角度:')
        arg_label_text.append('机器内外旋角 最小值(+内旋,-外旋):')
        arg_label_text.append('机器内外旋角 区间长度:')
        arg_label_text.append('机器内外旋角 步进角度:')
        arg_label_text.append('俯仰眼位 夹角[-30,+30]度:')
        arg_label_text.append('鼻颞眼位 夹角[-30,+30]度:')

    # Create entry widgets for each argument using a for loop
    for i, arg_text in enumerate(arg_label_text):
        arg_label = tk.Label(root, text=arg_text)
        arg_entry = tk.Entry(root)
        entries[arg_text] = arg_entry

        # Position the widgets using the grid layout
        arg_label.grid(row=i, column=0, padx=5, pady=5, sticky="e")
        arg_entry.grid(row=i, column=1, padx=5, pady=5)
    return entries

# Open the first window
open_first_window()

# Start the Tkinter event loop
tk.mainloop()
