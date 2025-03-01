import numpy as np
import napari
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QDoubleSpinBox,
    QCheckBox, QPushButton, QFrame
)
from qtpy.QtCore import Qt


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