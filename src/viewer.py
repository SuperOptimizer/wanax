import numpy as np
import napari
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QDoubleSpinBox,
    QCheckBox, QPushButton, QFrame
)
from qtpy.QtCore import Qt



class VolumeViewer:
    def __init__(self, volume: np.ndarray):
        if not isinstance(volume, np.ndarray) or volume.dtype != np.uint8:
            raise ValueError("Volume must be a uint8 numpy array")

        self.volume = volume
        self.viewer = napari.Viewer(ndisplay=3)
        self._add_volume()
        self._add_control_dock()

    def _add_volume(self):
        self.layer = self.viewer.add_image(
            self.volume,
            name='Volume',
            rendering='attenuated_mip',
            attenuation=0.1,
            contrast_limits=[0, 255],
            gamma=1.0,
        )

    def _add_control_dock(self):
        self.control_dock = VolumeControls("asdf",self.layer)
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

        # Attenuation control
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

        for vol in volumes:
            if not isinstance(vol, np.ndarray) or vol.dtype != np.uint8:
                raise ValueError("All volumes must be uint8 numpy arrays")

        self.volumes = volumes
        self.names = names
        self.viewer = napari.Viewer(ndisplay=3)
        self._add_volumes()
        self._add_control_dock()

    def _add_volumes(self):
        self.layers = []
        for volume, name in zip(self.volumes, self.names):
            layer = self.viewer.add_image(
                volume,
                name=name,
                rendering='attenuated_mip',
                attenuation=0.1,
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

