import tkinter as tk
import numpy as np
from tkinter import filedialog
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import mujoco
import mujoco.viewer

class CurveCreatorApp:
    def __init__(self, root, threshold=65, total_points=600):
        self.i = 0
        self.spline_points = None
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
        
        # Load model and initialize
        model = mujoco.MjModel.from_xml_path("scene.xml")
        nq = model.nq
        self.curve_names = [model.joint(i).name for i in range(nq)]
        self.limits = [(model.joint(i).range[0], model.joint(i).range[1]) for i in range(nq)]
        self.colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for _ in range(nq)]
        
        self.curve_visibility = [False for _ in range(len(self.curve_names))]
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
        
        self.clipped_spline_points = clipped_spline_points
        # self.clipped_spline_points = self.emotion_from(clipped_spline_points)
        
        # sample the spline points to match the total_points
        for i, spline in enumerate(self.clipped_spline_points):
            if len(spline) > 0:
                x = np.array([point[0] for point in spline])
                y = np.array([point[1] for point in spline])
                x_new = np.linspace(x.min(), x.max(), self.total_points)
                y_new = np.interp(x_new, x, y)
                self.clipped_spline_points[i] = [[x_new[j], y_new[j]] for j in range(len(x_new))]
        
        self.walk_cycle_time = self.walk_cycle_length.get()

        # save clipped spline points to a npy file
        np.save('clipped_spline_points.npy', np.array(self.clipped_spline_points))

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
        self.curves = [[(0, self.canvas_height // 2, 0, self.canvas_height // 2, 0, self.canvas_height // 2), (self.canvas_width, self.canvas_height // 2, self.canvas_width, self.canvas_height // 2, self.canvas_width, self.canvas_height // 2)] for _ in range(len(self.curve_names))]
        self.spline_points = [self.generate_bezier_curve(curve, self.canvas_height, self.limits[i], i, self.total_points) for i, curve in enumerate(self.curves)]
        self.draw_curve()
        self.dragging_point_index = None
        print('Gait reset')
        
    def on_key_press(self, event):
        # up
        if event.char.lower() == 'w' or event.keysym == 'Up':
            self.move_forward()
            
    def get_inertial_force(self):
        acceleration = self.accelerometer.getValues()
        
        return self.mass * acceleration[2]
    
    def calculate_moments(self, contact_points, force, com):
        M_x = 0
        M_y = 0
        for point in contact_points:
            point = point.getPoint()
            
            dx = point[0] - com[0]
            dy = point[1] - com[1]
            
            M_x += dy * force
            M_y += -dx * force 
        
        return M_x, M_y

    def compute_zmp(self, com, M_x, M_y, F_z):
        """
        Compute Zero Moment Point assuming z is vertical.
        com: (x_com, y_com, z_com)
        M_x, M_y: torque about CoM around x and y axes
        F_z: net vertical force
        Returns: (zmp_x, zmp_y)
        """
        x_com = com[0]
        y_com = com[1]
        
        # Classic 2D formula for ZMP
        zmp_x = x_com - (M_y / F_z)
        zmp_y = y_com + (M_x / F_z)
        return (zmp_x, zmp_y)

            
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

        # Update robot motor positions with interpolation
        timestep = self.sup.getBasicTimeStep()
        step_fraction = max(1, int(timestep / (self.walk_cycle_time * 1000) * len(trajectory)))

        for sub_step in range(step_fraction):
            # Interpolate positions for smoother transition
            for idx, position in enumerate(trajectory):
                if position is not None:
                    motor = self.sup.getDevice(self.curve_names[idx])
                    current_position = motor.getPositionSensor().getValue()
                    interpolated_position = current_position + (position - current_position) / step_fraction
                    motor.setPosition(interpolated_position)

            # Wait for the robot to stabilize for this sub-step
            self.sup.step(int(timestep / step_fraction))
            
            # Check balance
            balanced = self.node.getStaticBalance()
            contact_points = self.node.getContactPoints(includeDescendants=True)
            com = self.node.getCenterOfMass()
            inertial_force = self.get_inertial_force()
            M_x, M_y = self.calculate_moments(contact_points, inertial_force, com)
            zmp_x, zmp_y = self.compute_zmp(com, M_x, M_y, inertial_force)

            print('Balanced' if balanced else 'Not balanced\n', 'Mx: ', M_x, 'My: ', M_y)
            print(f"ZMP => x: {zmp_x:.3f}, y: {zmp_y:.3f}")

        # Update index safely
        longest_curve = max(len(spline) for spline in self.clipped_spline_points if spline)
        if longest_curve > 0:  # Avoid division by zero
            increment = max(1, int(timestep / (self.walk_cycle_time * 1000) * longest_curve))
            self.i = (self.i + increment) % longest_curve
        else:
            print("Longest curve is zero, cannot move forward.")


    def on_right_click(self, event):
        curve = self.curves[self.current_curve.get()]
        for i, (x, y, vector_1_x, vector_1_y, vector_2_x, vector_2_y) in enumerate(curve):
            if i != 0 and i != len(curve) - 1 and abs(x - event.x) < 10 and abs(y - event.y) < 10:
                del curve[i]  # Remove the point
                self.spline_points[self.current_curve.get()] = self.generate_bezier_curve(curve, self.canvas_height, self.limits[self.current_curve.get()], self.current_curve.get(), self.total_points)
                self.draw_curve()
                break

    def on_click(self, event):
        # If the selected curve is not visible, do not allow modification
        if not self.curve_visibility[self.current_curve.get()]:
            return
        
        if event.x < 0 or event.x > self.canvas_width or event.y < 0 or event.y > self.canvas_height:
            return
        
        event.x = float(event.x)
        event.y = float(event.y)
        
        # print(event.x, event.y)

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
            if y_to_angle(event.y, self.canvas_height) > self.limits[self.current_curve.get()][1] or y_to_angle(event.y, self.canvas_height) < self.limits[self.current_curve.get()][0]:
                return
            
            points = self.curves[self.current_curve.get()] + [(event.x, event.y, event.x + 60, event.y, event.x - 60, event.y)]
            points.sort(key=lambda p: p[0])
            self.dragging_point_index = (points.index((event.x, event.y, event.x + 60, event.y, event.x - 60, event.y)), 0)
        else: points = self.curves[self.current_curve.get()]
        
        for point in points:
            if y_to_angle(point[1], self.canvas_height) > self.limits[self.current_curve.get()][1] or y_to_angle(point[1], self.canvas_height) < self.limits[self.current_curve.get()][0]:
                return
        
        self.curves[self.current_curve.get()] = points
        self.spline_points[self.current_curve.get()] = self.generate_bezier_curve(points, self.canvas_height, self.limits[self.current_curve.get()], self.current_curve.get(), self.total_points)
        self.draw_curve()
        
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
            
    def import_motion_file(self):
        # Open a file dialog to select a .motion file
        file_path = filedialog.askopenfilename(
            title="Select a .motion file",
            filetypes=[("Motion files", "*.motion"), ("All files", "*.*")]
        )
        if file_path:
            self.upload_motion(file_path)

    def on_drag(self, event):
        orig_x = event.x
        orig_y = event.y
        
        event.x = float(event.x)
        event.y = float(event.y)
        
        # If the selected curve is not visible, do not allow modification
        if not self.curve_visibility[self.current_curve.get()]:
            return

        if self.dragging_point_index is not None:
            point, vector = self.dragging_point_index
            
            if event.x < 0 or event.x > self.canvas_width or event.y < 0 or event.y > self.canvas_height:
                return

            if vector == 2 and point == 0:
                vector = 1
            elif vector == 1 and point == len(self.curves[self.current_curve.get()]) - 1:
                vector = 2
            
            # if shift key is not held...
            cur = self.curves[self.current_curve.get()][point]
            # Assume 'point, vector' are your anchor + handle indices in self.dragging_point_index
            if vector == 0:
                new_y = event.y  # The mouse's vertical position

                if point == 0:
                    # --- LEFTMOST ANCHOR ---
                    # Pin X to 0 and preserve existing control-handle offsets
                    left_x, left_y, l_v1x, l_v1y, l_v2x, l_v2y = self.curves[self.current_curve.get()][0]
                    delta_y = new_y - left_y
                    
                    self.curves[self.current_curve.get()][0] = (
                        0,               # forced X = 0
                        new_y,           # new Y from mouse
                        l_v1x,           # keep original handle X
                        l_v1y + delta_y, # shift handle Y by the same delta
                        l_v2x,           # keep original handle X
                        l_v2y + delta_y  # shift handle Y by the same delta
                    )

                    # --- RIGHTMOST ANCHOR ---
                    # Must also share the same Y
                    rx, ry, rv1x, rv1y, rv2x, rv2y = self.curves[self.current_curve.get()][-1]
                    delta_y_r = new_y - ry
                    
                    self.curves[self.current_curve.get()][-1] = (
                        self.canvas_width, 
                        new_y,             # match the same new_y
                        rv1x,
                        rv1y + delta_y_r,
                        rv2x,
                        rv2y + delta_y_r
                    )

                elif point == len(self.curves[self.current_curve.get()]) - 1:
                    # --- RIGHTMOST ANCHOR ---
                    # Pin X to canvas_width, preserve control handles
                    rx, ry, rv1x, rv1y, rv2x, rv2y = self.curves[self.current_curve.get()][-1]
                    delta_y_r = new_y - ry
                    
                    self.curves[self.current_curve.get()][-1] = (
                        self.canvas_width,
                        new_y,
                        rv1x,
                        rv1y + delta_y_r,
                        rv2x,
                        rv2y + delta_y_r
                    )

                    # --- LEFTMOST ANCHOR ---
                    # Must share the same Y
                    lx, ly, lv1x, lv1y, lv2x, lv2y = self.curves[self.current_curve.get()][0]
                    delta_y_l = new_y - ly
                    
                    self.curves[self.current_curve.get()][0] = (
                        0,
                        new_y,
                        lv1x,
                        lv1y + delta_y_l,
                        lv2x,
                        lv2y + delta_y_l
                    )

                else:
                    # "cur" holds the old anchor + handle data: (x, y, v1x, v1y, v2x, v2y)
                    old_x, old_y, old_v1x, old_v1y, old_v2x, old_v2y = cur
                    
                    new_x = event.x
                    new_y = event.y
                    
                    # Calculate how far we are moving in X/Y
                    delta_x = new_x - old_x
                    delta_y = new_y - old_y
                    
                    # Move the anchor AND both of its control handles by the same delta
                    self.curves[self.current_curve.get()][point] = (
                        new_x,                # new anchor X
                        new_y,                # new anchor Y
                        old_v1x + delta_x,    # handle1 X
                        old_v1y + delta_y,    # handle1 Y
                        old_v2x + delta_x,    # handle2 X
                        old_v2y + delta_y     # handle2 Y
                    )



            else:
                if vector == 1:
                    self.curves[self.current_curve.get()][point] = (cur[0], cur[1], event.x, event.y, cur[4], cur[5])
                else:
                    self.curves[self.current_curve.get()][point] = (cur[0], cur[1], cur[2], cur[3], event.x, event.y)
                    
            curve = self.generate_bezier_curve(self.curves[self.current_curve.get()], self.canvas_height, self.limits[self.current_curve.get()], self.current_curve.get(), self.total_points)
            # check if curve overlaps with itself
            for i in range(len(curve) - 1):
                if curve[i][0] > curve[i + 1][0] or y_to_angle(curve[i][1], self.canvas_height) > self.limits[self.current_curve.get()][1] or y_to_angle(curve[i][1], self.canvas_height) < self.limits[self.current_curve.get()][0]:
                    self.curves[self.current_curve.get()][point] = cur
                    event.x = orig_x
                    event.y = orig_y
                    return
                
            self.spline_points[self.current_curve.get()] = curve
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
                for x, y, vector_1_x, vector_1_y, vector_2_x, vector_2_y in points:
                    self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color)
                    
                    self.canvas.create_line(x, y, vector_1_x, vector_1_y, fill=color, width=1)
                    self.canvas.create_oval(vector_1_x - 5, vector_1_y - 5, vector_1_x + 5, vector_1_y + 5, outline=color)
                    
                    self.canvas.create_line(x, y, vector_2_x, vector_2_y, fill=color, width=1)
                    self.canvas.create_oval(vector_2_x - 5, vector_2_y - 5, vector_2_x + 5, vector_2_y + 5, outline=color)

                    # Replace with something like this:
                    coords = []
                    for (x, y) in self.spline_points[curve_index]:
                        coords.extend([x, y])

                    # Draw the entire spline in one shot:
                    self.canvas.create_line(*coords, fill=color, width=2, smooth=True, splinesteps=36)

                    
    @staticmethod
    def generate_bezier_curve(points, canvas_height, limits, curve_index, total_points=1000):
        if len(points) < 2:
            return points

        bezier_points = []

        final_x, final_y, _, _, _, _ = points[-1]

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
                # if bezier_point[0] > final_x:
                #     bezier_point[0] = final_x
                bezier_points.append(bezier_point)

        filtered_points = bezier_points
        # filtered_points = []
        # last_x = float('-inf')  # Track the last x-coordinate
        # for bp in bezier_points:
        #     if bp[0] >= last_x:
        #         filtered_points.append(bp)
        #         last_x = bp[0]

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