import tkinter as tk
import time
import threading
import numpy as np
from tkinter import filedialog
from controller import Robot, Motor

class CurveCreatorApp:
    def __init__(self, root, threshold=65, total_points=600):
        self.i = 0
        self.spline_points = None
        self.robot = Robot()
        self.clipped_spline_points = None
        
        self.root = root
        self.root.title("Curve Creator")
        
        self.emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
        
        self.threshold = threshold
        self.total_points = total_points
        self.current_curve = tk.IntVar(value=0)  # Index of the current selected curve (starts with 0)

        self.walk_cycle_length = tk.DoubleVar(value=1.0)
        self.walk_cycle_time = self.walk_cycle_length.get()

        # Canvas dimensions
        self.canvas_width = 800
        self.canvas_height = 400
        self.center_line_y = self.canvas_height // 2

        # Initialize canvas
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white", bd=2, relief="solid")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Draw initial horizontal line
        self.canvas.create_line(0, self.center_line_y, self.canvas_width, self.center_line_y, fill="gray", dash=(4, 2))

        # Event bindings
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.root.bind("<Key>", self.on_key_press)
        
        ts = int(self.robot.getBasicTimeStep())
        self.curve_names = []
        self.limits = []
        self.colors = []
        for i in range(self.robot.getNumberOfDevices()):
            device = self.robot.getDeviceByIndex(i)
            if isinstance(device, Motor):
                self.curve_names.append(device.getName())
                self.limits.append((device.getMinPosition(), device.getMaxPosition()))
                self.colors.append('#%06X' % np.random.randint(0, 0xFFFFFF))
                device.getPositionSensor().enable(ts)
        self.robot.step(ts)
        
        self.curve_visibility = [True for _ in range(len(self.curve_names))]
        self.reset_gait()
        self.controls_frame = None
        self.create_ui()

    def create_ui(self):
        if self.controls_frame:
            self.controls_frame.destroy()
        
        # Create a frame to hold the controls
        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.X)

        # Create a frame for the Walk Cycle Length control
        walk_cycle_frame = tk.Frame(self.controls_frame)
        walk_cycle_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        tk.Label(walk_cycle_frame, text="Walk Cycle Length (s):").pack(side=tk.LEFT, padx=5)
        tk.Entry(walk_cycle_frame, textvariable=self.walk_cycle_length, width=10).pack(side=tk.LEFT, padx=5)
        
        # Add "Update Gait" button below the Walk Cycle Length control
        update_gait_button = tk.Button(self.controls_frame, text="Update Gait", command=self.update_gait)
        update_gait_button.pack(side=tk.TOP, pady=10)
        
        # Add "Reset Gait" button below the Update Gait button
        reset_gait_button = tk.Button(self.controls_frame, text="Reset Gait", command=self.reset_gait)
        reset_gait_button.pack(side=tk.TOP, pady=10)

        # Add "Import .proto file" button
        # import_proto_button = tk.Button(self.controls_frame, text="Import .proto file", command=self.import_proto_file)
        # import_proto_button.pack(side=tk.TOP, pady=10)

        # Add "Import .motion file" button below the Reset Gait button
        import_motion_button = tk.Button(self.controls_frame, text="Import .motion file", command=self.import_motion_file)
        import_motion_button.pack(side=tk.TOP, pady=10)
        
        emotions_frame = tk.Frame(self.controls_frame, bd=2, relief='solid')
        emotions_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        self.emotion_sliders = {}
        for emotion in self.emotions:
            emotion_slider = tk.Scale(emotions_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, length=200, label=emotion)
            emotion_slider.pack(side=tk.LEFT, padx=5)
            self.emotion_sliders[emotion] = emotion_slider

        # Create a frame to hold the scrollable region
        curve_controls_frame = tk.Frame(self.controls_frame, bd=2, relief='solid')
        curve_controls_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create a canvas widget for the scrollable area
        self.scrollable_canvas = tk.Canvas(curve_controls_frame, width=250)
        self.scrollable_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a scrollbar for the scrollable canvas
        scrollbar = tk.Scrollbar(curve_controls_frame, orient="vertical", command=self.scrollable_canvas.yview)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        # Configure the scrollable canvas to use the scrollbar
        self.scrollable_canvas.configure(yscrollcommand=scrollbar.set)

        # Create a frame inside the canvas to hold the radio buttons
        self.radio_frame = tk.Frame(self.scrollable_canvas)
        self.scrollable_canvas.create_window((0, 0), window=self.radio_frame, anchor="nw")

        # Add radio buttons for selecting curves
        self.curve_selector_label = tk.Label(self.radio_frame, text="Select Curve:")
        self.curve_selector_label.grid(row=0, column=0, padx=5)

        for i in range(len(self.curve_names)):
            # Create a radio button for each curve with its descriptive name
            radio_button = tk.Radiobutton(self.radio_frame, text=self.curve_names[i], variable=self.current_curve, value=i,
                                        command=self.update_selected_curve, fg=self.colors[i])
            radio_button.grid(row=i + 1, column=0, padx=5)

        # Add checkboxes for visibility control
        self.visibility_label = tk.Label(self.radio_frame, text="Toggle Curve Visibility:")
        self.visibility_label.grid(row=0, column=1, padx=5)

        self.checkbuttons = []
        for i in range(len(self.curve_names)):
            var = tk.BooleanVar(value=self.curve_visibility[i])  # Use BooleanVar to track checkbox state
            check_button = tk.Checkbutton(self.radio_frame, text=self.curve_names[i], variable=var, 
                                        command=self.toggle_visibility, fg=self.colors[i])
            check_button.var = var  # Store the BooleanVar in the widget
            check_button.grid(row=i + 1, column=1, padx=5)
            self.checkbuttons.append(check_button)

        # Update the scrollable area to reflect the correct size
        self.radio_frame.update_idletasks()
        self.scrollable_canvas.config(scrollregion=self.scrollable_canvas.bbox("all"))

    def toggle_visibility(self):
        # Update the visibility list based on the checkbox states
        for i, check_button in enumerate(self.checkbuttons):
            self.curve_visibility[i] = check_button.var.get()
        self.draw_curve()

    def update_selected_curve(self):
        self.draw_curve()

    def update_gait(self):
        clipped_spline_points = []
        for i, spline in enumerate(self.spline_points):
            if len(spline) > 0:
                transformed = np.array(spline.copy(), dtype=np.float64)
                transformed = abs(transformed - self.canvas_height)
                transformed /= float(self.canvas_height)
                transformed *= 2 * np.pi
                transformed -= np.pi
                
                lower, upper = self.limits[i]
                clipped = np.clip(transformed, lower, upper)
                clipped = [[spline[j][0], clipped[j][1]] for j in range(len(clipped))]
                
                clipped_spline_points.append(clipped)
            else:
                clipped_spline_points.append(np.zeros((self.total_points, 2)).tolist())
                
        # only keep those that are <= self.canvas_width
        for i, spline in enumerate(clipped_spline_points):
            clipped_spline_points[i] = [point for point in spline if point[0] <= self.canvas_width]
        
        self.clipped_spline_points = self.emotion_from(clipped_spline_points)
        
        self.walk_cycle_time = self.walk_cycle_length.get()

        print("Gait saved")
        

    def emotion_from(self, spline_points):
        emotion_values = {}
        for emotion in self.emotion_sliders:
            emotion_values[emotion] = self.emotion_sliders[emotion].get()
        
        # Define emotion effects with realistic scales
        effects = {
            "Anger": {"hip_pitch": emotion_values['Anger'], "shoulder_pitch": emotion_values['Anger'], "head_pitch": -emotion_values['Anger'], "elbow_roll": emotion_values['Fear']},
            "Disgust": {"hip_pitch": -emotion_values['Disgust'], "shoulder_pitch": -emotion_values['Disgust']},
            "Fear": {"head_pitch": emotion_values['Fear'], "hip_pitch": -emotion_values['Fear'], "shoulder_pitch": -emotion_values['Fear'], "elbow_roll": emotion_values['Fear']},
            "Happiness": {"hip_pitch": emotion_values['Happiness'], "shoulder_pitch": emotion_values['Happiness'], "head_pitch": -emotion_values['Happiness']},
            "Sadness": {"hip_pitch": -emotion_values['Sadness'], "shoulder_pitch": -emotion_values['Sadness'], "head_pitch": emotion_values['Sadness']},
            "Surprise": {"hip_pitch": -emotion_values['Surprise'], "head_yaw": emotion_values['Surprise'], "head_pitch": -emotion_values['Surprise']},
        }
        
        l_hip_pitch_idx = self.curve_names.index('LHipPitch')
        r_hip_pitch_idx = self.curve_names.index('RHipPitch')
        l_shoulder_pitch_idx = self.curve_names.index('LShoulderPitch')
        r_shoulder_pitch_idx = self.curve_names.index('RShoulderPitch')
        head_pitch_idx = self.curve_names.index('HeadPitch')
        head_yaw_idx = self.curve_names.index('HeadYaw')
        r_elbow_roll_idx = self.curve_names.index('RElbowRoll')
        l_elbow_roll_idx = self.curve_names.index('LElbowRoll')
        mean_roll = effects['Anger']['elbow_roll'] / 2 + effects['Fear']['elbow_roll'] / 2
        for i in range(len(spline_points)):
            mean = np.mean([point[1] for point in spline_points[i]])
            for j in range(len(spline_points[i])):
                if i == l_hip_pitch_idx:
                    total_effect = 1/6 * (effects['Anger']['hip_pitch'] + effects['Disgust']['hip_pitch'] + effects['Fear']['hip_pitch'] + effects['Happiness']['hip_pitch'] + effects['Sadness']['hip_pitch'] + effects['Surprise']['hip_pitch'])
                    spline_points[i][j][1] = self.add_if_gt_mean(spline_points[i][j][1], total_effect, mean)
                elif i == r_hip_pitch_idx:
                    total_effect = 1/6 * (effects['Anger']['hip_pitch'] + effects['Disgust']['hip_pitch'] + effects['Fear']['hip_pitch'] + effects['Happiness']['hip_pitch'] + effects['Sadness']['hip_pitch'] + effects['Surprise']['hip_pitch'])
                    spline_points[i][j][1] = self.add_if_gt_mean(spline_points[i][j][1], total_effect, mean)
                elif i == l_shoulder_pitch_idx:
                    total_effect = 1/5 * (effects['Anger']['shoulder_pitch'] + effects['Disgust']['shoulder_pitch'] + effects['Fear']['shoulder_pitch'] + effects['Happiness']['shoulder_pitch'] + effects['Sadness']['shoulder_pitch'])
                    spline_points[i][j][1] = self.add_if_gt_mean(spline_points[i][j][1], total_effect, mean)
                elif i == r_shoulder_pitch_idx:
                    total_effect = 1/5 * (effects['Anger']['shoulder_pitch'] + effects['Disgust']['shoulder_pitch'] + effects['Fear']['shoulder_pitch'] + effects['Happiness']['shoulder_pitch'] + effects['Sadness']['shoulder_pitch'])
                    spline_points[i][j][1] = self.add_if_gt_mean(spline_points[i][j][1], total_effect, mean)
                elif i == head_pitch_idx:
                    spline_points[i][j][1] += effects['Anger']['head_pitch'] / 4
                    spline_points[i][j][1] += effects['Fear']['head_pitch'] / 4
                    spline_points[i][j][1] += effects['Sadness']['head_pitch'] / 2
                elif i == head_yaw_idx:
                    spline_points[i][j][1] += effects['Surprise']['head_yaw']
                elif i == r_elbow_roll_idx and (effects['Anger']['elbow_roll'] != 0 or effects['Fear']['elbow_roll'] != 0):
                    spline_points[i][j][1] += (effects['Anger']['elbow_roll'] + effects['Fear']['elbow_roll']) / 2 - mean_roll + np.pi / 2
                elif i == l_elbow_roll_idx and (effects['Anger']['elbow_roll'] != 0 or effects['Fear']['elbow_roll'] != 0):
                    spline_points[i][j][1] = -((effects['Anger']['elbow_roll'] + effects['Fear']['elbow_roll']) / 2 - mean_roll + np.pi / 2)
                    
        speedup = emotion_values['Anger'] * 0.25
        slowdown = emotion_values['Sadness'] * 0.25
        self.walk_cycle_length.set(self.walk_cycle_length.get() + slowdown - speedup)
                    
        return spline_points
    
    def add_if_gt_mean(self, value, to_add, mean):
        if value < mean:
            return value - to_add
        else:
            return value + to_add

    
    def reset_gait(self): # sets all curves to 0
        self.curves = [[] for _ in range(len(self.curve_names))]
        self.spline_points = [[] for _ in range(len(self.curve_names))]
        self.draw_curve()
        self.dragging_point_index = None
        print('Gait reset')
        
    def on_key_press(self, event):
        # up
        if event.char.lower() == 'w' or event.keysym == 'Up':
            self.move_forward()
            
    def move_forward(self):
        if not self.clipped_spline_points or len(self.clipped_spline_points) == 0:
            print("No spline points to move forward.")
            return

        trajectory = []
        for spline in self.clipped_spline_points:
            if spline and len(spline) > self.i:  # Ensure spline is not empty and index is valid
                trajectory.append(spline[self.i][1])
            else:
                trajectory.append(None)

        # Update robot motor positions
        for idx, position in enumerate(trajectory):
            if position is not None:
                motor = self.robot.getDevice(self.curve_names[idx])
                motor.setPosition(position)

        # Step the robot simulation
        ts = int(self.robot.getBasicTimeStep())
        self.robot.step(ts)

        # Update index safely
        longest_curve = max(len(spline) for spline in self.clipped_spline_points if spline)
        if longest_curve > 0:  # Avoid division by zero
            increment = int(ts / (self.walk_cycle_time * 1000) * longest_curve)
            self.i = (self.i + increment) % longest_curve
        else:
            print("Longest curve is zero, cannot move forward.")


    def on_right_click(self, event):
        curve = self.curves[self.current_curve.get()]
        for i, (x, y, vector_1_x, vector_1_y, vector_2_x, vector_2_y) in enumerate(curve):
            if abs(x - event.x) < 10 and abs(y - event.y) < 10:
                del curve[i]  # Remove the point
                self.draw_curve()
                break

    def on_click(self, event):
        # If the selected curve is not visible, do not allow modification
        if not self.curve_visibility[self.current_curve.get()]:
            return
        
        if event.x < 0 or event.x > self.canvas_width or event.y < 0 or event.y > self.canvas_height:
            return
        
        if y_to_angle(event.y, self.canvas_height) > self.limits[self.current_curve.get()][1] or y_to_angle(event.y, self.canvas_height) < self.limits[self.current_curve.get()][0]:
            return
        
        event.x = float(event.x)
        event.y = float(event.y)

        # Check if clicking near an existing point
        for i, (x, y, vector_1_x, vector_1_y, vector_2_x, vector_2_y) in enumerate(self.curves[self.current_curve.get()]):
            if (abs(x - event.x) < 10 and abs(y - event.y) < 10) and not (event.state & 0x1):
                self.dragging_point_index = (i, 0)
                return
            elif (abs(vector_1_x - event.x) < 10 and abs(vector_1_y - event.y) < 10) and (event.state & 0x1):
                self.dragging_point_index = (i, 1)
                return
            elif (abs(vector_2_x - event.x) < 10 and abs(vector_2_y - event.y) < 10) and (event.state & 0x1):
                self.dragging_point_index = (i, 2)
                return

        # Add a new point after the closest existing point
        if not (event.state & 0x1):
            points = self.curves[self.current_curve.get()] + [(event.x, event.y, event.x, event.y, event.x, event.y)]
            points.sort(key=lambda p: p[0])
        
        self.dragging_point_index = (points.index((event.x, event.y, event.x, event.y, event.x, event.y)), 0)
        
        self.curves[self.current_curve.get()] = points
        self.draw_curve()
        
    def upload_proto(self, proto_file):
        self.curve_names = []
        self.colors = []
        self.limits = []
        
        # reset curves using the function reset_gait
        self.reset_gait()
        
        with open(proto_file, 'r') as f:
            lines = f.readlines()
            f.close()
        
        for i in range(len(lines)):
            if 'RotationalMotor' in lines[i]:
                name = lines[i + 1].lstrip().split(' ')[1].strip('\n').strip('"')
                self.curve_names.append(name)
                
                llimit = lines[i + 3].lstrip().split(' ')[1]
                hlimit = lines[i + 4].lstrip().split(' ')[1]
                
                self.limits.append((float(llimit), float(hlimit)))
                
                # add a random color
                self.colors.append('#%06X' % np.random.randint(0, 0xFFFFFF))
                
                self.curves.append([])
                self.spline_points.append([])

        ts = int(self.robot.getBasicTimeStep())
        sensors = []
        for name in self.curve_names:
            sensor = self.robot.getDevice(name + 'S')
            sensor.enable(ts)
            sensors.append(sensor)
        self.robot.step(ts)

        self.curve_visibility = [True] * len(self.curve_names)
        self.draw_curve()

        # Create UI elements for curve selection and visibility toggling
        self.create_ui()
        
    def upload_motion(self, motion_file):
        with open(motion_file, 'r') as f:
            lines = f.readlines()
            f.close()
            
        lines = [line for line in lines if line.strip()]
        file_idx_to_curve_idx = {}
        num_lines = len(lines) - 1
        for i in range(0, len(lines), int(len(lines) / 13)):
            line = lines[i]
            if line.startswith("#WEBOTS_MOTION"):
                curve_names = line.split(",")[2:]
                for j, curve_name in enumerate(curve_names):
                    curve_name = curve_name.strip('\n')
                    if curve_name in self.curve_names:
                        file_idx_to_curve_idx[j] = self.curve_names.index(curve_name)
                    else:
                        print(curve_name)
            else:
                positions = [float(pos) for pos in line.split(',')[2:]]
                for j in range(len(positions)):
                    self.curves[file_idx_to_curve_idx[j]].append((i / num_lines * self.canvas_width, angle_to_y(positions[j], self.canvas_height)))
                
        line = lines[-1]
        first_line = lines[1]
        start_t = first_line.split(',')[0]
        start_t = start_t.split(':')
        start_t = int(start_t[0]) * 60 + int(start_t[1]) + int(start_t[2]) / 1000
        t = line.split(',')[0]
        t = t.split(':')
        t = int(t[0]) * 60 + int(t[1]) + int(t[2]) / 1000 - start_t
        self.walk_cycle_length.set(t)
        
        self.draw_curve()
            
    def import_proto_file(self):
        # Open a file dialog to select a .proto file
        file_path = filedialog.askopenfilename(
            title="Select a .proto file",
            filetypes=[("Proto files", "*.proto"), ("All files", "*.*")]
        )
        if file_path:
            self.upload_proto(file_path)
            
    def import_motion_file(self):
        # Open a file dialog to select a .motion file
        file_path = filedialog.askopenfilename(
            title="Select a .motion file",
            filetypes=[("Motion files", "*.motion"), ("All files", "*.*")]
        )
        if file_path:
            self.upload_motion(file_path)

    def on_drag(self, event):
        event.x = float(event.x)
        event.y = float(event.y)
        
        # If the selected curve is not visible, do not allow modification
        if not self.curve_visibility[self.current_curve.get()]:
            return
        
        point, vector = self.dragging_point_index

        if self.dragging_point_index is not None:
            if event.x < 0 or event.x > self.canvas_width or event.y < 0 or event.y > self.canvas_height:
                return

            if vector == 2 and point == 0:
                vector = 1
            elif vector == 1 and point == len(self.curves[self.current_curve.get()]) - 1:
                vector = 2
            
            # if shift key is not held...
            cur = self.curves[self.current_curve.get()][point]
            if vector == 0:
                if point > 0 and event.x <= self.curves[self.current_curve.get()][point -  1][0] + self.threshold:
                    return
                if point < len(self.curves[self.current_curve.get()]) - 1 and event.x >= self.curves[self.current_curve.get()][point + 1][0] - self.threshold:
                    return
                if y_to_angle(event.y, self.canvas_height) > self.limits[self.current_curve.get()][1] or y_to_angle(event.y, self.canvas_height) < self.limits[self.current_curve.get()][0]:
                    return
                
                if point == 0:
                    self.curves[self.current_curve.get()][point] = (event.x, event.y, cur[2], cur[3], event.x, event.y)
                elif point == len(self.curves[self.current_curve.get()]) - 1:
                    self.curves[self.current_curve.get()][point] = (event.x, event.y, event.x, event.y, cur[4], cur[5])
                else:
                    self.curves[self.current_curve.get()][point] = (event.x, event.y, cur[2], cur[3], cur[4], cur[5])
            else:
                if vector == 1:
                    self.curves[self.current_curve.get()][point] = (cur[0], cur[1], event.x, event.y, cur[4], cur[5])
                else:
                    self.curves[self.current_curve.get()][point] = (cur[0], cur[1], cur[2], cur[3], event.x, event.y)
                
            self.draw_curve()

    def on_release(self, event):
        self.dragging_point_index = None

    def draw_curve(self):
        # Clear the canvas and redraw the horizontal line
        self.canvas.delete("all")
        self.canvas.create_line(0, self.center_line_y, self.canvas_width, self.center_line_y, fill="gray", dash=(4, 2))

        # Draw all the curves that are visible
        for curve_index in range(len(self.curve_names)):
            if self.curve_visibility[curve_index]:  # Only draw if visible
                color = self.colors[curve_index]
                # Draw points for the current curve
                points = self.curves[curve_index].copy()
                # if len(self.curves[curve_index]) > 0:
                #     last_x, last_y, last_vector_x, last_vector_y = self.curves[curve_index][-1]
                #     first_x, first_y, first_vector_x, first_vector_y = self.curves[curve_index][0]
                #     dist_end = self.canvas_width - last_x
                #     dist_start = first_x
                #     new_y = last_y + (first_y - last_y) * (dist_end / (dist_end + dist_start))
                #     if last_x < self.canvas_width:
                #         points.append((self.canvas_width, new_y, self.canvas_width, new_y))
                #     if first_x > 0:
                #         points.insert(0, (0, new_y, 0, new_y))
                for x, y, vector_1_x, vector_1_y, vector_2_x, vector_2_y in points:
                    self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color)
                    
                    self.canvas.create_line(x, y, vector_1_x, vector_1_y, fill=color, width=1)
                    self.canvas.create_oval(vector_1_x - 5, vector_1_y - 5, vector_1_x + 5, vector_1_y + 5, outline=color)
                    
                    self.canvas.create_line(x, y, vector_2_x, vector_2_y, fill=color, width=1)
                    self.canvas.create_oval(vector_2_x - 5, vector_2_y - 5, vector_2_x + 5, vector_2_y + 5, outline=color)

                # Draw the curve using Bezier Curve
                self.spline_points[curve_index] = self.generate_bezier_curve(points, self.canvas_height, self.limits, curve_index, self.total_points)
                for i in range(len(self.spline_points[curve_index]) - 1):
                    x1, y1 = self.spline_points[curve_index][i]
                    x2, y2 = self.spline_points[curve_index][i + 1]
                    self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2)
                    
    @staticmethod
    def generate_bezier_curve(points, canvas_height, limits, curve_index, total_points=1000):
        if len(points) < 2:
            return points

        bezier_points = []

        for i in range(len(points) - 1):
            # Extract anchor points and control points
            p0 = np.array([points[i][0], points[i][1]])     # Anchor i
            c1 = np.array([points[i][2], points[i][3]])     # Control 1 at anchor i
            c2 = np.array([points[i + 1][4], points[i + 1][5]])  # Control 2 at anchor i+1
            p3 = np.array([points[i + 1][0], points[i + 1][1]])  # Anchor i+1

            # Generate points along the cubic Bézier curve
            for t in np.linspace(0, 1, total_points):
                bezier_point = (
                    (1 - t)**3 * p0 +
                    3 * (1 - t)**2 * t * c1 +
                    3 * (1 - t) * t**2 * c2 +
                    t**3 * p3
                )
                bezier_points.append(bezier_point)

        # -----------------------------------------------------------------
        # FILTER OUT ANY POINTS THAT MOVE BACKWARD IN X (OVERLAPPING FIX)
        # -----------------------------------------------------------------
        filtered_points = []
        last_x = float('-inf')  # Track the last x-coordinate
        for bp in bezier_points:
            if bp[0] > last_x:
                filtered_points.append(bp)
                last_x = bp[0]
        # Now we have a strictly increasing sequence of x-coordinates

        return np.array(filtered_points)


    
def y_to_angle(y, canvas_height):
    moved = canvas_height - y
    normalized = moved / canvas_height
    angle = (normalized * 2 - 1) * np.pi
    return angle

def angle_to_y(angle, canvas_height):
    normalized = (angle / np.pi) / 2 + 0.5
    moved = normalized * canvas_height
    y = canvas_height - moved
    return y

if __name__ == "__main__":
    root = tk.Tk()
    app = CurveCreatorApp(root)

    root.mainloop()