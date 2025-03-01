import numpy as np
import napari
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QDoubleSpinBox,
    QCheckBox, QPushButton, QFrame
)
from qtpy.QtCore import Qt
from typing import List, Tuple


class ChannelControls(QWidget):
    def __init__(self, name, layer, color, parent=None):
        super().__init__(parent)
        self.name = name
        self.layer = layer
        self.color = color

        layout = QVBoxLayout()
        self.setLayout(layout)

        header = QLabel(f"<h3 style='color:{color}'>{name}</h3>")
        layout.addWidget(header)

        # Attenuation control - greatly extended range
        att_layout = QHBoxLayout()
        att_layout.addWidget(QLabel("Attenuation:"))

        self.att_slider = QSlider(Qt.Horizontal)
        self.att_slider.setMinimum(1)  # 0.0001 minimum (1/10000)
        self.att_slider.setMaximum(100000)  # 10.0 maximum (100000/10000)
        self.att_slider.setValue(int(layer.attenuation * 10000))
        self.att_slider.valueChanged.connect(self.update_attenuation)
        att_layout.addWidget(self.att_slider)

        self.att_value = QDoubleSpinBox()
        self.att_value.setRange(0.0001, 10.0)
        self.att_value.setDecimals(4)  # Show 4 decimal places
        self.att_value.setValue(layer.attenuation)
        self.att_value.setSingleStep(0.01)
        self.att_value.valueChanged.connect(self.update_attenuation_from_spinbox)
        att_layout.addWidget(self.att_value)

        layout.addLayout(att_layout)

        # Gamma control - greatly extended range
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma:"))

        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setMinimum(1)  # 0.0001 minimum
        self.gamma_slider.setMaximum(100000)  # 10.0 maximum
        self.gamma_slider.setValue(int(layer.gamma * 10000))
        self.gamma_slider.valueChanged.connect(self.update_gamma)
        gamma_layout.addWidget(self.gamma_slider)

        self.gamma_value = QDoubleSpinBox()
        self.gamma_value.setRange(0.0001, 10.0)
        self.gamma_value.setDecimals(4)  # Show 4 decimal places
        self.gamma_value.setValue(layer.gamma)
        self.gamma_value.setSingleStep(0.01)
        self.gamma_value.valueChanged.connect(self.update_gamma_from_spinbox)
        gamma_layout.addWidget(self.gamma_value)

        layout.addLayout(gamma_layout)

        # Visibility toggle
        vis_layout = QHBoxLayout()
        vis_layout.addWidget(QLabel("Visible:"))
        self.vis_checkbox = QCheckBox()
        self.vis_checkbox.setChecked(layer.visible)
        self.vis_checkbox.stateChanged.connect(self.toggle_visibility)
        vis_layout.addWidget(self.vis_checkbox)

        layout.addLayout(vis_layout)

    def update_attenuation(self, value):
        att_value = value / 10000.0  # Convert from slider value to actual value
        self.layer.attenuation = att_value
        self.att_value.blockSignals(True)
        self.att_value.setValue(att_value)
        self.att_value.blockSignals(False)

    def update_attenuation_from_spinbox(self, value):
        self.layer.attenuation = value
        self.att_slider.blockSignals(True)
        self.att_slider.setValue(int(value * 10000))
        self.att_slider.blockSignals(False)

    def update_gamma(self, value):
        gamma_value = value / 10000.0  # Convert from slider value to actual value
        self.layer.gamma = gamma_value
        self.gamma_value.blockSignals(True)
        self.gamma_value.setValue(gamma_value)
        self.gamma_value.blockSignals(False)

    def update_gamma_from_spinbox(self, value):
        self.layer.gamma = value
        self.gamma_slider.blockSignals(True)
        self.gamma_slider.setValue(int(value * 10000))
        self.gamma_slider.blockSignals(False)

    def toggle_visibility(self, state):
        self.layer.visible = bool(state)


class VolumeControlsDock(QWidget):
    def __init__(self, layers, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Volume Controls")

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.channel_controls = []
        colors = ['red', 'green', 'blue']
        names = ['Red Channel', 'Green Channel', 'Blue Channel']

        for i, (layer, color, name) in enumerate(zip(layers, colors, names)):
            controls = ChannelControls(name, layer, color)
            layout.addWidget(controls)
            self.channel_controls.append(controls)

            if i < len(layers) - 1:
                separator = QFrame()
                separator.setFrameShape(QFrame.HLine)
                separator.setFrameShadow(QFrame.Sunken)
                layout.addWidget(separator)

        global_layout = QVBoxLayout()
        global_label = QLabel("<h3>Global Settings</h3>")
        global_layout.addWidget(global_label)

        reset_button = QPushButton("Reset All")
        reset_button.clicked.connect(self.reset_all)
        global_layout.addWidget(reset_button)

        layout.addLayout(global_layout)
        layout.addStretch()

    def reset_all(self):
        for control in self.channel_controls:
            control.att_value.setValue(0.1)
            control.gamma_value.setValue(1.0)
            control.vis_checkbox.setChecked(True)

class VolumeViewer:
    def __init__(self, volume: np.ndarray):
        if not isinstance(volume, np.ndarray):
            raise ValueError("Volume must be a numpy array")

        # Convert to uint8 if needed
        if volume.dtype != np.uint8:
            print(f"Converting volume from {volume.dtype} to uint8")
            # Scale to 0-255 range if not already in that range
            if volume.min() < 0 or volume.max() > 255:
                volume = ((volume - volume.min()) / (volume.max() - volume.min()) * 255).astype(np.uint8)
            else:
                volume = volume.astype(np.uint8)

        # Check for all-zero data which could cause division issues
        if np.all(volume == 0):
            print("Warning: Volume contains all zeros. Adding a small value to prevent division issues.")
            volume[0, 0, 0] = 1  # Add a single non-zero value

        self.volume = volume
        self.viewer = napari.Viewer(ndisplay=3)
        self._add_volume()
        self._add_control_dock()

    def _add_volume(self):
        try:
            self.layer = self.viewer.add_image(
                self.volume,
                name='Volume',
                rendering='attenuated_mip',
                attenuation=0.1,
                contrast_limits=[0, 255],
                gamma=1.0,
            )
        except Exception as e:
            print(f"Error adding volume: {e}")
            # Try with different rendering mode as fallback
            self.layer = self.viewer.add_image(
                self.volume,
                name='Volume',
                rendering='mip',  # Try maximum intensity projection instead
                contrast_limits=[0, 255],
                gamma=1.0,
            )

    def _add_control_dock(self):
        self.control_dock = VolumeControls("Volume Controls", self.layer)
        self.viewer.window.add_dock_widget(
            self.control_dock,
            name="Volume Controls",
            area="right"
        )

    def run(self):
        napari.run()


class VolumeControls(QWidget):
    def __init__(self, name, layer, parent=None):
        super().__init__(parent)
        self.layer = layer

        layout = QVBoxLayout()
        self.setLayout(layout)

        header = QLabel(f"<h3>{name}</h3>")
        layout.addWidget(header)

        # Attenuation control (only if the layer has attenuation property)
        if hasattr(layer, 'attenuation'):
            att_layout = QHBoxLayout()
            att_layout.addWidget(QLabel("Attenuation:"))

            self.att_slider = QSlider(Qt.Horizontal)
            self.att_slider.setMinimum(1)
            self.att_slider.setMaximum(100000)
            self.att_slider.setValue(int(layer.attenuation * 10000))
            self.att_slider.valueChanged.connect(self.update_attenuation)
            att_layout.addWidget(self.att_slider)

            self.att_value = QDoubleSpinBox()
            self.att_value.setRange(0.0001, 10.0)
            self.att_value.setDecimals(4)
            self.att_value.setValue(layer.attenuation)
            self.att_value.setSingleStep(0.01)
            self.att_value.valueChanged.connect(self.update_attenuation_from_spinbox)
            att_layout.addWidget(self.att_value)

            layout.addLayout(att_layout)

        # Gamma control
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma:"))

        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setMinimum(1)
        self.gamma_slider.setMaximum(100000)
        self.gamma_slider.setValue(int(layer.gamma * 10000))
        self.gamma_slider.valueChanged.connect(self.update_gamma)
        gamma_layout.addWidget(self.gamma_slider)

        self.gamma_value = QDoubleSpinBox()
        self.gamma_value.setRange(0.0001, 10.0)
        self.gamma_value.setDecimals(4)
        self.gamma_value.setValue(layer.gamma)
        self.gamma_value.setSingleStep(0.01)
        self.gamma_value.valueChanged.connect(self.update_gamma_from_spinbox)
        gamma_layout.addWidget(self.gamma_value)

        layout.addLayout(gamma_layout)

        # Visibility toggle
        vis_layout = QHBoxLayout()
        vis_layout.addWidget(QLabel("Visible:"))
        self.vis_checkbox = QCheckBox()
        self.vis_checkbox.setChecked(layer.visible)
        self.vis_checkbox.stateChanged.connect(self.toggle_visibility)
        vis_layout.addWidget(self.vis_checkbox)

        layout.addLayout(vis_layout)

    def update_attenuation(self, value):
        att_value = value / 10000.0
        self.layer.attenuation = att_value
        self.att_value.blockSignals(True)
        self.att_value.setValue(att_value)
        self.att_value.blockSignals(False)

    def update_attenuation_from_spinbox(self, value):
        self.layer.attenuation = value
        self.att_slider.blockSignals(True)
        self.att_slider.setValue(int(value * 10000))
        self.att_slider.blockSignals(False)

    def update_gamma(self, value):
        gamma_value = value / 10000.0
        self.layer.gamma = gamma_value
        self.gamma_value.blockSignals(True)
        self.gamma_value.setValue(gamma_value)
        self.gamma_value.blockSignals(False)

    def update_gamma_from_spinbox(self, value):
        self.layer.gamma = value
        self.gamma_slider.blockSignals(True)
        self.gamma_slider.setValue(int(value * 10000))
        self.gamma_slider.blockSignals(False)

    def toggle_visibility(self, state):
        self.layer.visible = bool(state)


class MultiVolumeControlsDock(QWidget):
    def __init__(self, layers, names, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.volume_controls = []
        for i, (layer, name) in enumerate(zip(layers, names)):
            controls = VolumeControls(name, layer)
            layout.addWidget(controls)
            self.volume_controls.append(controls)

            if i < len(layers) - 1:
                separator = QFrame()
                separator.setFrameShape(QFrame.HLine)
                separator.setFrameShadow(QFrame.Sunken)
                layout.addWidget(separator)

        reset_button = QPushButton("Reset All")
        reset_button.clicked.connect(self.reset_all)
        layout.addWidget(reset_button)
        layout.addStretch()

    def reset_all(self):
        for control in self.volume_controls:
            if hasattr(control.layer, 'attenuation'):
                control.att_value.setValue(0.1)
            control.gamma_value.setValue(1.0)
            control.vis_checkbox.setChecked(True)


class MultiVolumeViewer:
    def __init__(self, volumes, names=None):
        if not isinstance(volumes, list):
            volumes = [volumes]

        if names is None:
            names = [f'Volume {i+1}' for i in range(len(volumes))]
        elif len(names) != len(volumes):
            raise ValueError("Number of names must match number of volumes")

        self.volumes = []
        for vol in volumes:
            if not isinstance(vol, np.ndarray):
                raise ValueError("All volumes must be numpy arrays")

            # Convert to uint8 if needed
            if vol.dtype != np.uint8:
                print(f"Converting volume from {vol.dtype} to uint8")
                # Scale to 0-255 range if not already in that range
                if vol.min() < 0 or vol.max() > 255:
                    vol = ((vol - vol.min()) / (vol.max() - vol.min()) * 255).astype(np.uint8)
                else:
                    vol = vol.astype(np.uint8)

            # Check for all-zero data which could cause division issues
            if np.all(vol == 0):
                print("Warning: Volume contains all zeros. Adding a small value to prevent division issues.")
                vol[0, 0, 0] = 1  # Add a single non-zero value

            self.volumes.append(vol)

        self.names = names
        self.viewer = napari.Viewer(ndisplay=3)
        self._add_volumes()
        self._add_control_dock()

    def _add_volumes(self):
        self.layers = []
        for volume, name in zip(self.volumes, self.names):
            try:
                layer = self.viewer.add_image(
                    volume,
                    name=name,
                    rendering='attenuated_mip',
                    attenuation=0.1,
                    contrast_limits=[0, 255],
                    gamma=1.0,
                    blending='additive'
                )
            except Exception as e:
                print(f"Error adding volume {name}: {e}")
                # Try with different rendering mode as fallback
                layer = self.viewer.add_image(
                    volume,
                    name=name,
                    rendering='mip',  # Try maximum intensity projection instead
                    contrast_limits=[0, 255],
                    gamma=1.0,
                    blending='additive'
                )

            self.layers.append(layer)

    def _add_control_dock(self):
        self.control_dock = MultiVolumeControlsDock(self.layers, self.names)
        self.viewer.window.add_dock_widget(
            self.control_dock,
            name="Volume Controls",
            area="right"
        )

    def run(self):
        napari.run()




class SimpleRGBVolumeViewer:
    def __init__(self,
                 arrays: List[str],
                 start_coords: Tuple[int, int, int] = (0, 0, 0),
                 size_coords: Tuple[int, int, int] = None):
        if len(arrays) != 3:
            raise ValueError("Need exactly three zarr paths")

        print(f"Input data type: {arrays[0].dtype}")
        self.shape = arrays[0].shape
        self.start = start_coords

        if size_coords is None:
            self.size = self.shape
            self.size = self.size[0], self.size[1], self.size[2]
        else:
            self.size = size_coords

        self.channels = self._load_channels(arrays)
        self.viewer = napari.Viewer(ndisplay=3)
        self._add_color_channels()
        self._add_control_dock()

    def _load_channels(self, arrays):
        channels = []
        for i, arr in enumerate(arrays):
            chunk = arr[
                    self.start[0]:self.start[0] + self.size[0],
                    self.start[1]:self.start[1] + self.size[1],
                    self.start[2]:self.start[2] + self.size[2]
                    ]

            channels.append(chunk)
            print(
                f"Channel {i} stats after conversion: min={np.min(chunk)}, max={np.max(chunk)}, mean={np.mean(chunk):.2f}")
        return channels

    def _add_color_channels(self):
        colors = ['red', 'green', 'blue']
        names = ['Red Channel', 'Green Channel', 'Blue Channel']

        self.layers = []

        for i, (channel, color, name) in enumerate(zip(self.channels, colors, names)):
            layer = self.viewer.add_image(
                channel,
                name=name,
                colormap=color,
                blending='additive',
                rendering='attenuated_mip',
                attenuation=0.1,
                contrast_limits=[0, 255],
                gamma=1.0,
            )
            self.layers.append(layer)

        print(f"Added 3 separate channels with individual coloring")

    def _add_control_dock(self):
        self.control_dock = VolumeControlsDock(self.layers)
        self.viewer.window.add_dock_widget(
            self.control_dock,
            name="Volume Controls",
            area="right"
        )

    def run(self):
        print(f"Starting viewer with 3 colored channels and custom controls")
        napari.run()