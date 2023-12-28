import tkinter as tk
from tkinter import messagebox

from enum import Enum

class ResultMode(Enum):
    Single = 1
    Loop = 2

class LensMode(Enum):
    Lens = 1
    FrontBoard = 2

INTERACTIVE_INPUT = True  # Set this to False if you want to use default values

def process_input(entries, arg_label_text, root, next_window, run_mesh, *previous_args):
    try:
        # Extract values from the entries
        args = [entry.get() for entry in entries.values()]

        arg_dict = {}
        # Print or use the values
        for i, (text,value) in enumerate(zip(arg_label_text, args)):
            arg_dict.update({text:value})
            print(f"{text}: {value}")
        print('\n')
        # Call your function or perform the desired operation with the inputs here

        if run_mesh:
            result_mode_int = int(previous_args[0])
            lens_mode_int = int(previous_args[1])

            result_mode = ResultMode(result_mode_int)
            lens_mode = LensMode(lens_mode_int)

            working_distance = arg_dict['工作距离(周边)[0,45]mm']

            if lens_mode == LensMode.FrontBoard:
                front_board_name = arg_dict['面板模型文件名(无后缀)']
            if lens_mode == LensMode.Lens:
                lens_diameter = arg_dict['目镜外框直径[20,80]mm']
                cone_diameter = arg_dict['通光孔径[20,80]mm']
                cone_angle = arg_dict['单张范围（眼外角）[70,140]度']
            if result_mode == ResultMode.Single:
                lens_alpha = arg_dict['机器俯仰角[-90,90]度(+仰,-俯)']
                lens_beta = arg_dict['机器内外旋角[-90,90]度(+内旋,-外旋)']
                eye_alpha = arg_dict['眼睛俯仰角[-90,90]度(+俯,-仰)']
                eye_beta = arg_dict['眼睛内外旋角[-90,90]度(+外旋,-内旋)']
            if result_mode == ResultMode.Loop:
                lens_alpha_min = arg_dict['机器俯仰角 最小值(+仰,-俯)']
                lens_alpha_range = arg_dict['机器俯仰角 区间长度']
                lens_alpha_stride = arg_dict['机器俯仰角 步进角度']
                lens_beta_min = arg_dict['机器内外旋角 最小值(+内旋,-外旋)']
                lens_beta_range = arg_dict['机器内外旋角 区间长度']
                lens_beta_stride = arg_dict['机器内外旋角 步进角度']
                side_alpha = arg_dict['俯仰眼位 夹角[-30,+30]度']
                side_beta = arg_dict['鼻颞眼位 夹角[-30,+30]度']

            print('\n')
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
    entries, arg_label_text = create_first_window_input_fields(first_window)

    # Create a button that calls the process_input function when clicked
    button_process = tk.Button(first_window, text="Process Input", command=lambda: process_input(entries, arg_label_text, first_window, open_second_window, False, *args))
    button_process.grid(row=2, column=0, columnspan=2, pady=10)

def open_second_window(*args):
    second_window = tk.Tk()
    second_window.title("选择工作参数")

    # Create input fields and get the entries dictionary for the second window
    entries, arg_label_text = create_second_window_input_fields(second_window, *args)

    # Create a button that calls the process_input function when clicked
    button_process = tk.Button(second_window, text="Process Input", command=lambda: process_input(entries, arg_label_text, second_window, lambda *args: open_first_window(*args), True, *args))
    button_process.grid(row=15, column=0, columnspan=2, pady=10)

def create_first_window_input_fields(root):
    # Create a dictionary to store the entry widgets
    entries = {}

    arg_label_text = ['选择模式(1:单组参数 2:遍历参数)', '选择模式(1:镜片碰撞/遮挡 2:面板碰撞)']

    # Create entry widgets for each argument using a for loop
    for i, arg_text in enumerate(arg_label_text):
        arg_label = tk.Label(root, text=arg_text)
        arg_entry = tk.Entry(root)
        entries[arg_text] = arg_entry

        # Position the widgets using the grid layout
        arg_label.grid(row=i, column=0, padx=5, pady=5, sticky="e")
        arg_entry.grid(row=i, column=1, padx=5, pady=5)
    return entries, arg_label_text

def create_second_window_input_fields(root, *args):
    # Create a dictionary to store the entry widgets
    entries = {}

    arg_label_text = ['工作距离(周边)[0,45]mm']

    # Use arg1 to determine the number and labels for entry widgets
    if args[1] == '2':
        arg_label_text.append('面板模型文件名(无后缀)')
    if args[1] == '1':
        arg_label_text.append('目镜外框直径[20,80]mm')
        arg_label_text.append('通光孔径[20,80]mm')
        arg_label_text.append('单张范围（眼外角）[70,140]度')

    if args[0] == '1':
        arg_label_text.append('机器俯仰角[-90,90]度(+仰,-俯)')
        arg_label_text.append('机器内外旋角[-90,90]度(+内旋,-外旋)')
        arg_label_text.append('眼睛俯仰角[-90,90]度(+俯,-仰)')
        arg_label_text.append('眼睛内外旋角[-90,90]度(+外旋,-内旋)')
    if args[0] == '2':
        arg_label_text.append('机器俯仰角 最小值(+仰,-俯)')
        arg_label_text.append('机器俯仰角 区间长度')
        arg_label_text.append('机器俯仰角 步进角度')
        arg_label_text.append('机器内外旋角 最小值(+内旋,-外旋)')
        arg_label_text.append('机器内外旋角 区间长度')
        arg_label_text.append('机器内外旋角 步进角度')
        arg_label_text.append('俯仰眼位 夹角[-30,+30]度')
        arg_label_text.append('鼻颞眼位 夹角[-30,+30]度')

    # Create entry widgets for each argument using a for loop
    for i, arg_text in enumerate(arg_label_text):
        arg_label = tk.Label(root, text=arg_text)
        arg_entry = tk.Entry(root)
        entries[arg_text] = arg_entry

        # Position the widgets using the grid layout
        arg_label.grid(row=i, column=0, padx=5, pady=5, sticky="e")
        arg_entry.grid(row=i, column=1, padx=5, pady=5)
    return entries, arg_label_text

# Open the first window
open_first_window()

# Start the Tkinter event loop
tk.mainloop()
