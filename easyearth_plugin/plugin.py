from qgis.PyQt.QtWidgets import (QAction, QDockWidget, QPushButton, QVBoxLayout, 
                                QWidget, QMessageBox, QLabel, QHBoxLayout,
                                QLineEdit, QFileDialog, QComboBox, QGroupBox, QGridLayout, QInputDialog, QProgressBar, QCheckBox, QButtonGroup, QRadioButton)
from qgis.PyQt.QtCore import Qt, QByteArray, QBuffer, QIODevice, QProcess, QTimer, QProcessEnvironment, QVariant
from qgis.PyQt.QtGui import QIcon, QMovie, QColor
from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry, QgsPolygon, 
                      QgsPointXY, QgsField, QgsProject, QgsPoint, QgsLineString,
                      QgsWkbTypes, QgsRasterLayer, Qgis, QgsApplication, QgsVectorFileWriter, QgsSymbol, QgsCategorizedSymbolRenderer, 
                      QgsRendererCategory, QgsMarkerSymbol, QgsFillSymbol, QgsCoordinateTransform, QgsSingleSymbolRenderer, QgsFields)
from qgis.gui import QgsMapToolEmitPoint, QgsRubberBand
import os
import requests
import base64
from PIL import Image
import io
import numpy as np
import subprocess
import signal
import time
import logging
import shutil
import tempfile
import sys
import yaml
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class EasyEarthPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.actions = []
        self.menu = 'EasyEarth'
        self.toolbar = self.iface.addToolBar(u'EasyEarth')
        self.toolbar.setObjectName(u'EasyEarth')
        
        # Initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        
        # Docker configuration
        self.project_name = "easyearth_plugin"
        self.service_name = self.get_service_name()
        self.image_name = f"{self.project_name}_{self.service_name}"
        self.sudo_password = None  # Add this to store password temporarily
        
        # Initialize map tools and data
        self.canvas = iface.mapCanvas()
        self.point_tool = None
        self.points = []
        self.rubber_bands = []
        self.docker_process = None
        self.server_process = None
        self.server_port = 3781  # Default port
        self.server_url = f"http://0.0.0.0:{self.server_port}/v1/easyearth"
        self.current_image_path = None
        self.current_embedding_path = None
        self.docker_running = False
        self.server_running = False
        self.action = None
        self.dock_widget = None
        self.rubber_band = None
        self.is_selecting_points = False
        self.point_counter = None
        self.point_layer = None
        self.total_steps = 0
        self.current_step = 0
        self.is_drawing = False
        self.draw_tool = None
        self.temp_rubber_band = None
        self.start_point = None
        self.drawn_features = []
        self.temp_vector_path = os.path.join(tempfile.gettempdir(), 'drawn_features.gpkg')
        self.drawn_layer = None
        self.temp_geojson_path = os.path.join(tempfile.gettempdir(), 'drawn_features.geojson')
        self.feature_count = 0  # For generating unique IDs
        self.temp_prompts_geojson = None
        self.temp_predictions_geojson = None
        self.real_time_prediction = False
        self.prompts_layer = None
        self.predictions_layer = None
        logger.debug("Plugin initialized")

        # Initialize map tool
        self.map_tool = QgsMapToolEmitPoint(self.canvas)
        self.map_tool.canvasClicked.connect(self.handle_draw_click)
        
        # Initialize rubber bands
        self.rubber_band = QgsRubberBand(self.canvas, QgsWkbTypes.PointGeometry)
        self.rubber_band.setColor(QColor(255, 0, 0))
        self.rubber_band.setWidth(2)
        
        self.temp_rubber_band = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
        self.temp_rubber_band.setColor(QColor(255, 0, 0, 50))
        self.temp_rubber_band.setWidth(2)
        
        self.start_point = None
        self.predictions_geojson = None
        self.predictions_layer = None

    def add_action(self, icon_path, text, callback, enabled_flag=True,
                  add_to_menu=True, add_to_toolbar=True, status_tip=None,
                  whats_this=None, parent=None):
        """Add a toolbar icon to the toolbar"""
        
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(self.menu, action)

        self.actions.append(action)
        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI"""
        try:
            logger.debug("Starting initGui")
            
            # Create dock widget
            self.dock_widget = QDockWidget('EasyEarth Plugin', self.iface.mainWindow())
            self.dock_widget.setObjectName('EasyEarthPluginDock')
            
            # Create main widget and layout
            main_widget = QWidget()
            main_layout = QVBoxLayout()

            # 1. Docker Control Group
            docker_group = QGroupBox("Docker Control")
            docker_layout = QVBoxLayout()

            # Docker status and button layout
            status_layout = QHBoxLayout()
            docker_label = QLabel("Docker Status:")
            self.docker_status = QLabel("Stopped")
            self.docker_button = QPushButton("Start Docker")
            self.docker_button.clicked.connect(self.toggle_docker)

            status_layout.addWidget(docker_label)
            status_layout.addWidget(self.docker_status)
            status_layout.addWidget(self.docker_button)

            # Add progress bar and progress status
            self.progress_bar = QProgressBar()
            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(100)
            self.progress_bar.hide()
            
            self.progress_status = QLabel()
            self.progress_status.setWordWrap(True)
            self.progress_status.hide()
            
            docker_layout.addLayout(status_layout)
            docker_layout.addWidget(self.progress_bar)
            docker_layout.addWidget(self.progress_status)

            docker_group.setLayout(docker_layout)
            main_layout.addWidget(docker_group)

            # 2. Service Information Group
            service_group = QGroupBox("Service Information")
            service_layout = QVBoxLayout()

            # Server status
            status_layout = QHBoxLayout()
            server_label = QLabel("Server Status:")
            self.server_status = QLabel("Checking...")
            status_layout.addWidget(server_label)
            status_layout.addWidget(self.server_status)
            status_layout.addStretch()
            service_layout.addLayout(status_layout)

            # API Information
            api_layout = QVBoxLayout()
            api_label = QLabel("API Endpoints:")
            api_label.setStyleSheet("font-weight: bold;")
            self.api_info = QLabel(f"Base URL: http://0.0.0.0:{self.server_port}/v1/easyearth\n"
                                  f"Predict: /sam-predict\n"
                                  f"Health: /ping")
            self.api_info.setWordWrap(True)
            api_layout.addWidget(api_label)
            api_layout.addWidget(self.api_info)
            service_layout.addLayout(api_layout)

            service_group.setLayout(service_layout)
            main_layout.addWidget(service_group)

            # 3. Image Source Group
            image_group = QGroupBox("Image Source")
            image_layout = QVBoxLayout()
            
            # Image source selection
            source_layout = QHBoxLayout()
            source_label = QLabel("Source:")
            self.source_combo = QComboBox()
            self.source_combo.addItems(["File", "Layer"])
            self.source_combo.currentTextChanged.connect(self.on_image_source_changed)
            source_layout.addWidget(source_label)
            source_layout.addWidget(self.source_combo)
            image_layout.addLayout(source_layout)
            
            # File input
            file_layout = QHBoxLayout()
            self.image_path = QLineEdit()
            self.browse_button = QPushButton("Browse Image")
            self.browse_button.clicked.connect(self.browse_image)
            file_layout.addWidget(self.image_path)
            file_layout.addWidget(self.browse_button)
            image_layout.addLayout(file_layout)
            
            # Layer selection
            self.layer_combo = QComboBox()
            self.layer_combo.hide()
            image_layout.addWidget(self.layer_combo)

            image_group.setLayout(image_layout)
            main_layout.addWidget(image_group)

            # 4. Embedding Settings Group
            embedding_group = QGroupBox("Embedding Settings")
            embedding_layout = QVBoxLayout()

            # Radio buttons for embedding options
            self.embedding_options = QButtonGroup()
            self.no_embedding_radio = QRadioButton("No embedding file")
            self.load_embedding_radio = QRadioButton("Load existing embedding")
            self.save_embedding_radio = QRadioButton("Save new embedding")
            
            self.embedding_options.addButton(self.no_embedding_radio)
            self.embedding_options.addButton(self.load_embedding_radio)
            self.embedding_options.addButton(self.save_embedding_radio)
            
            # Set default option
            self.no_embedding_radio.setChecked(True)

            # Embedding path selection
            self.embedding_path_layout = QHBoxLayout()
            self.embedding_path_edit = QLineEdit()
            self.embedding_path_edit.setEnabled(False)
            self.embedding_browse_btn = QPushButton("Browse")
            self.embedding_browse_btn.setEnabled(False)
            self.embedding_browse_btn.clicked.connect(self.browse_embedding)
            
            self.embedding_path_layout.addWidget(self.embedding_path_edit)
            self.embedding_path_layout.addWidget(self.embedding_browse_btn)

            # Connect radio buttons to handler
            self.embedding_options.buttonClicked.connect(self.on_embedding_option_changed)

            # Add widgets to embedding layout
            embedding_layout.addWidget(self.no_embedding_radio)
            embedding_layout.addWidget(self.load_embedding_radio)
            embedding_layout.addWidget(self.save_embedding_radio)
            embedding_layout.addLayout(self.embedding_path_layout)

            embedding_group.setLayout(embedding_layout)
            main_layout.addWidget(embedding_group)

            # 5. Drawing and Prediction Settings Group
            settings_group = QGroupBox("Drawing and Prediction Settings")
            settings_layout = QVBoxLayout()

            # Drawing type selection
            type_layout = QHBoxLayout()
            type_label = QLabel("Draw type:")
            self.draw_type_combo = QComboBox()
            self.draw_type_combo.addItems(["Point", "Box", "Text"])
            self.draw_type_combo.setItemData(2, False, Qt.UserRole - 1)  # Disable Text option
            self.draw_type_combo.currentTextChanged.connect(self.on_draw_type_changed)
            type_layout.addWidget(type_label)
            type_layout.addWidget(self.draw_type_combo)
            settings_layout.addLayout(type_layout)

            # Drawing button
            self.draw_button = QPushButton("Start Drawing")
            self.draw_button.setCheckable(True)
            self.draw_button.clicked.connect(self.toggle_drawing)
            self.draw_button.setEnabled(False)  # Enabled after image is loaded
            settings_layout.addWidget(self.draw_button)

            settings_group.setLayout(settings_layout)
            main_layout.addWidget(settings_group)

            # Prediction Group
            predict_group = QGroupBox("Prediction")
            predict_layout = QVBoxLayout()
            self.predict_button = QPushButton("Get Prediction")
            self.predict_button.clicked.connect(self.get_prediction)
            predict_layout.addWidget(self.predict_button)
            predict_group.setLayout(predict_layout)
            main_layout.addWidget(predict_group)

            # Set the main layout
            main_widget.setLayout(main_layout)
            self.dock_widget.setWidget(main_widget)
            
            # Add dock widget to QGIS
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget)

            # Connect to project layer changes
            QgsProject.instance().layersAdded.connect(self.update_layer_combo)
            QgsProject.instance().layersRemoved.connect(self.update_layer_combo)

            # Connect to QGIS quit signal
            QgsApplication.instance().aboutToQuit.connect(self.cleanup_docker)

            # Start periodic server status check
            self.status_timer = QTimer()
            self.status_timer.timeout.connect(self.check_server_status)
            self.status_timer.start(5000)  # Check every 5 seconds

            logger.debug("Finished initGui setup")
        except Exception as e:
            logger.error(f"Error in initGui: {str(e)}")
            logger.exception("Full traceback:")

    def update_layer_combo(self):
        """Update the layers combo box with current raster layers"""
        self.layer_combo.clear()
        self.layer_combo.addItem("Select a layer...")
        
        # Add all raster layers to combo
        for layer in QgsProject.instance().mapLayers().values():
            if isinstance(layer, QgsRasterLayer):
                self.layer_combo.addItem(layer.name(), layer)

    def on_image_source_changed(self, text):
        """Handle image source selection change"""
        if text == "File":
            self.image_path.show()
            self.browse_button.show()
            self.layer_combo.hide()
        else:
            self.image_path.hide()
            self.browse_button.hide()
            self.layer_combo.show()
        
        if text != "File":
            self.update_layer_combo()

    def browse_image(self):
        """Open file dialog for image selection and load it to canvas"""
        try:
            # Set initial directory to plugin's data folder
            initial_dir = os.path.join(self.plugin_dir, 'data')
            if not os.path.exists(initial_dir):
                initial_dir = ""

            file_path, _ = QFileDialog.getOpenFileName(
                self.dock_widget,
                "Select Image File",
                initial_dir,
                "Image Files (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*.*)"
            )
            
            if file_path:
                self.image_path.setText(file_path)
                # Load image to canvas immediately
                self.load_image()
            
        except Exception as e:
            logger.error(f"Error browsing image: {str(e)}")
            QMessageBox.critical(None, "Error", f"Failed to browse image: {str(e)}")

    def load_image(self):
        """Load the selected image and create predictions layer"""
        try:
            # Get the image path
            image_path = self.image_path.text()
            if not image_path:
                return

            # Load the image as a raster layer
            raster_layer = QgsRasterLayer(image_path, "Selected Image")
            if raster_layer.isValid():
                # Add raster layer to the project
                QgsProject.instance().addMapLayer(raster_layer)
                
                # Create temporary file for predictions
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.predictions_geojson = os.path.join(
                    tempfile.gettempdir(), 
                    f'predictions_{timestamp}.geojson'
                )
                
                # Create empty GeoJSON file for predictions
                self.create_empty_geojson(self.predictions_geojson)
                
                # Create predictions layer
                self.predictions_layer = self.create_vector_layer(
                    self.predictions_geojson,
                    "SAM Predictions",
                    "prediction"
                )
                
                # Enable drawing controls
                self.draw_button.setEnabled(True)
                
                self.iface.messageBar().pushMessage(
                    "Success", 
                    "Image loaded and prediction layer created",
                    level=Qgis.Success,
                    duration=3
                )
            else:
                QMessageBox.warning(None, "Error", "Invalid raster layer")

        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            QMessageBox.critical(None, "Error", f"Failed to load image: {str(e)}")

    def create_vector_layer(self, geojson_path, layer_name, layer_type):
        """Create and style a vector layer with project CRS"""
        try:
            # Create vector layer
            layer = QgsVectorLayer(geojson_path, layer_name, "ogr")
            if not layer.isValid():
                raise Exception(f"Failed to create {layer_name} layer")
            
            # Set CRS to match project CRS
            project_crs = QgsProject.instance().crs()
            layer.setCrs(project_crs)
            
            # Add fields based on layer type
            provider = layer.dataProvider()
            if layer_type == "prompt":
                fields = QgsFields()
                fields.append(QgsField("id", QVariant.Int, "Integer"))
                fields.append(QgsField("type", QVariant.String, "String", 20))  # length 20 for "Point" or "Box"
                provider.addAttributes(fields)
            else:  # prediction
                fields = QgsFields()
                fields.append(QgsField("id", QVariant.Int, "Integer"))
                fields.append(QgsField("confidence", QVariant.Double, "Real", 10, 6))  # precision 6
                fields.append(QgsField("class", QVariant.String, "String", 50))  # length 50
                provider.addAttributes(fields)
            
            layer.updateFields()
            
            # Add to project first
            QgsProject.instance().addMapLayer(layer)
            
            # Apply styling after layer is added to project
            try:
                if layer_type == "prompt":
                    self.style_prompts_layer(layer)
                else:
                    self.style_predictions_layer(layer)
            except Exception as e:
                logger.error(f"Error applying style to layer: {str(e)}")
            
            return layer
            
        except Exception as e:
            logger.error(f"Error creating vector layer: {str(e)}")
            raise

    def style_prompts_layer(self, layer):
        """Style the prompts layer with different symbols for points and boxes"""
        try:
            # Create point symbol
            point_symbol = QgsMarkerSymbol.createSimple({
                'name': 'circle',
                'size': '3',
                'color': '255,0,0,255'  # Red
            })

            # Create box symbol
            box_symbol = QgsFillSymbol.createSimple({
                'color': '255,255,0,50',  # Semi-transparent yellow
                'outline_color': '255,0,0,255',  # Red outline
                'outline_width': '0.8'
            })

            # Create categories
            categories = [
                QgsRendererCategory('Point', point_symbol, 'Point'),
                QgsRendererCategory('Box', box_symbol, 'Box')
            ]

            # Create and apply the renderer
            renderer = QgsCategorizedSymbolRenderer('type', categories)
            layer.setRenderer(renderer)
            layer.triggerRepaint()

        except Exception as e:
            logger.error(f"Error styling prompts layer: {str(e)}")
            # Don't raise the exception - just log it and continue
            # This prevents the 'NoneType' error from stopping the layer creation

    def style_predictions_layer(self, layer):
        """Style the predictions layer"""
        try:
            # Create a simple fill symbol for predictions
            symbol = QgsFillSymbol.createSimple({
                'color': '0,255,0,50',  # Semi-transparent green
                'outline_color': '0,255,0,255',  # Green outline
                'outline_width': '0.8'
            })

            # Create and apply the renderer
            renderer = QgsSingleSymbolRenderer(symbol)
            layer.setRenderer(renderer)
            layer.triggerRepaint()

        except Exception as e:
            logger.error(f"Error styling predictions layer: {str(e)}")
            # Don't raise the exception - just log it and continue

    def clear_points(self):
        """Clear all selected points"""
        self.points = []
        if hasattr(self, 'point_counter'):
            self.point_counter.setText("Points: 0")
        if self.point_layer:
            try:
                # Remove features from the point layer
                self.point_layer.dataProvider().truncate()
                self.point_layer.triggerRepaint()
            except Exception as e:
                logger.error(f"Error clearing points: {str(e)}")

    def run(self):
        """Run method that loads and starts the plugin"""
        if self.dock_widget.isVisible():
            self.dock_widget.hide()
        else:
            self.dock_widget.show()

    def unload(self):
        """Cleanup when unloading the plugin"""
        try:
            # Clean up temporary files and layers
            self.cleanup_previous_session()
            
            # Remove the plugin menu item and icon
            if self.toolbar:
                self.toolbar.deleteLater()
            for action in self.actions:
                self.iface.removePluginMenu("Easy Earth", action)
                self.iface.removeToolBarIcon(action)
            
            # Clean up Docker resources
            self.cleanup_docker()
            
            # Clear sudo password
            self.sudo_password = None
            
            # Stop the status check timer
            if hasattr(self, 'status_timer'):
                self.status_timer.stop()
                del self.status_timer
            
            # Clear points
            self.clear_points()
            
            # Remove the plugin UI elements
            if self.dock_widget:
                self.iface.removeDockWidget(self.dock_widget)
            
            # Clean up any other resources
            if hasattr(self, 'point_layer') and self.point_layer:
                QgsProject.instance().removeMapLayer(self.point_layer.id())
            
            # Remove temporary drawn features layer
            if hasattr(self, 'drawn_layer') and self.drawn_layer:
                QgsProject.instance().removeMapLayer(self.drawn_layer.id())
            
            # Remove temporary file
            if os.path.exists(self.temp_geojson_path):
                os.remove(self.temp_geojson_path)
            
            # Remove layers
            if self.prompts_layer:
                QgsProject.instance().removeMapLayer(self.prompts_layer.id())
            if self.predictions_layer:
                QgsProject.instance().removeMapLayer(self.predictions_layer.id())
            
            # Remove temporary files
            for file_path in [self.temp_prompts_geojson, self.temp_predictions_geojson]:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            
            logger.debug("Plugin unloaded successfully")
        except Exception as e:
            logger.error(f"Error during plugin unload: {str(e)}")
            logger.exception("Full traceback:")

    def get_sudo_password(self):
        """Get sudo password if not already stored"""
        if not self.sudo_password:
            password, ok = QInputDialog.getText(None, 
                "Sudo Password Required", 
                "Enter sudo password:",
                QLineEdit.Password)
            if ok and password:
                self.sudo_password = password
                return password
            return None
        return self.sudo_password

    def run_sudo_command(self, cmd):
        """Run a command with sudo"""
        try:
            password = self.get_sudo_password()
            if not password:
                return None
            
            full_cmd = f'echo "{password}" | sudo -S {cmd}'
            return subprocess.run(['bash', '-c', full_cmd], capture_output=True, text=True)
        except Exception as e:
            logger.error(f"Error running sudo command: {str(e)}")
            return None

    def get_service_name(self):
        """Get the service name from docker-compose.yml"""
        try:
            compose_path = os.path.join(self.plugin_dir, 'docker-compose.yml')
            with open(compose_path, 'r') as file:
                compose_data = yaml.safe_load(file)
                # Get the first service name from the services dictionary
                service_name = next(iter(compose_data.get('services', {})))
                logger.debug(f"Found service name: {service_name}")
                return service_name
        except Exception as e:
            logger.error(f"Error getting service name: {str(e)}")
            return "easyearth-server"  # fallback default

    def check_docker_image(self):
        """Check if Docker image exists"""
        try:
            result = self.run_sudo_command(f"docker images {self.image_name}:latest -q")
            return bool(result and result.stdout.strip())
        except Exception as e:
            logger.error(f"Error checking Docker image: {str(e)}")
            return False

    def check_docker_running(self):
        """Check if Docker daemon is running"""
        try:
            result = self.run_sudo_command("docker info")
            return bool(result and result.returncode == 0)
        except Exception as e:
            logger.error(f"Error checking Docker status: {str(e)}")
            return False
    
    def check_docker_container_running(self):
        """Check if Docker container is running"""
        try:
            result = self.run_sudo_command("docker ps")
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error checking Docker container status: {str(e)}")
            return False
        
    def toggle_docker(self):
        """Toggle Docker container state"""
        try:
            # TODO: need to deal with the case where the docker container is initilized outside qgis, so the docker_running is actually true
            # TODO: need to test the server status right after the docker container is finished starting
            if not self.docker_running:
                # Check if Docker daemon is running
                if not self.check_docker_running():
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Critical)
                    msg.setText("Docker daemon is not running")
                    msg.setInformativeText("Please start Docker using one of these methods:\n\n"
                                        "1. Start Docker Desktop (if installed)\n\n"
                                        "2. Use command line:\n"
                                        "   systemctl start docker (Linux)\n"
                                        "   open -a Docker (macOS)\n\n"
                                        "After starting Docker, try again.")
                    msg.setWindowTitle("Docker Error")
                    msg.exec_()
                    return
                else:
                    # Get password at the start
                    if not self.get_sudo_password():
                        return

                    compose_path = os.path.join(self.plugin_dir, 'docker-compose.yml')
                    
                    # Initialize progress tracking
                    self.current_step = 0
                    self.count_docker_steps()
                    self.progress_bar.setValue(0)
                    self.progress_bar.show()
                    self.progress_status.show()
                    
                    self.docker_process = QProcess()
                    self.docker_process.finished.connect(self.on_docker_finished)
                    self.docker_process.errorOccurred.connect(self.on_docker_error)
                    self.docker_process.readyReadStandardError.connect(self.on_docker_stderr)
                    self.docker_process.readyReadStandardOutput.connect(self.on_docker_stdout)
                    
                    self.docker_process.setWorkingDirectory(os.path.dirname(compose_path))
                    
                    # Check if we need to build
                    if not self.check_docker_image():
                        cmd = f'echo "{self.sudo_password}" | sudo -S docker-compose -p {self.project_name} -f "{compose_path}" up -d --build'
                        self.progress_status.setText("Building Docker image (this may take a while)...")
                    else:
                        cmd = f'echo "{self.sudo_password}" | sudo -S docker-compose -p {self.project_name} -f "{compose_path}" up -d'
                        self.progress_status.setText("Starting existing Docker container...")
                    
                    self.docker_process.start('bash', ['-c', cmd])
                    self.docker_status.setText("Starting")
                    self.docker_button.setEnabled(False)
                    
            else:
                if not self.get_sudo_password():
                    return

                compose_path = os.path.join(self.plugin_dir, 'docker-compose.yml')
                cmd = f'echo "{self.sudo_password}" | sudo -S docker-compose -p {self.project_name} -f "{compose_path}" down'
                
                # Create new QProcess instance for stopping
                self.docker_process = QProcess()
                self.docker_process.finished.connect(self.on_docker_finished)
                self.docker_process.errorOccurred.connect(self.on_docker_error)
                self.docker_process.setWorkingDirectory(os.path.dirname(compose_path))
                
                self.docker_process.start('bash', ['-c', cmd])
                self.docker_status.setText("Stopping")
                self.docker_button.setEnabled(False)
                self.progress_status.setText("Stopping Docker container...")
                self.progress_bar.show()

        except Exception as e:
            logger.error(f"Error controlling Docker: {str(e)}")
            QMessageBox.critical(None, "Error", f"Error controlling Docker: {str(e)}")

    def on_docker_stderr(self):
        """Handle Docker process stderr output"""
        error = self.docker_process.readAllStandardError().data().decode()
        logger.error(f"Docker error: {error}")
        self._process_docker_output(error)

    def on_docker_stdout(self):
        """Handle Docker process stdout output"""
        output = self.docker_process.readAllStandardOutput().data().decode()
        logger.debug(f"Docker output: {output}")
        self._process_docker_output(output)

    def on_docker_finished(self, exit_code, exit_status):
        """Handle Docker process completion"""
        try:
            if exit_code == 0:
                self.progress_bar.setValue(self.total_steps)  # Set to 100%
                if not self.docker_running:
                    self.docker_running = True
                    self.docker_status.setText("Running")
                    self.docker_button.setText("Stop Docker")
                    self.progress_status.setText("Docker started successfully")
                    # Trigger immediate server status check
                    self.check_server_status()
                else:
                    self.docker_running = False
                    self.docker_status.setText("Stopped")
                    self.docker_button.setText("Start Docker")
                    self.progress_status.setText("Docker stopped successfully")
                    self.server_status.setText("Not Running")
                    self.server_status.setStyleSheet("color: red;")
            else:
                error_output = self.docker_process.readAllStandardError().data().decode()
                logger.error(f"Docker command failed with exit code {exit_code}. Error: {error_output}")
                QMessageBox.critical(None, "Error", 
                    f"Docker command failed with exit code {exit_code}")
                self.docker_status.setText("Error")
                self.progress_status.setText(f"Error: Docker command failed with exit code {exit_code}")
        finally:
            self.docker_button.setEnabled(True)
            QTimer.singleShot(2000, lambda: self.progress_bar.hide())
            QTimer.singleShot(2000, lambda: self.progress_status.hide())

    def on_docker_error(self, error):
        """Handle Docker process errors"""
        error_msg = {
            QProcess.FailedToStart: "Failed to start Docker process",
            QProcess.Crashed: "Docker process crashed",
            QProcess.Timedout: "Docker process timed out",
            QProcess.WriteError: "Write error occurred",
            QProcess.ReadError: "Read error occurred",
            QProcess.UnknownError: "Unknown error occurred"
        }.get(error, "An error occurred")
        
        logger.error(f"Docker error: {error_msg}")
        QMessageBox.critical(None, "Error", f"Docker error: {error_msg}")
        self.docker_status.setText("Error")
        self.docker_button.setEnabled(True)

    def toggle_drawing(self, checked):
        """Toggle drawing mode"""
        try:
            logger.info(f"Toggle drawing called with checked={checked}")
            
            if checked:
                logger.info("Starting new drawing session")
                
                # Remove existing temporary files and layers before creating new ones
                logger.info("Cleaning up previous session")
                self.cleanup_previous_session()
                
                # Create new temporary files and layers
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.temp_prompts_geojson = os.path.join(
                    tempfile.gettempdir(), 
                    f'prompts_{timestamp}.geojson'
                )
                self.temp_predictions_geojson = os.path.join(
                    tempfile.gettempdir(), 
                    f'predictions_{timestamp}.geojson'
                )
                
                logger.info(f"Created temporary file paths: \n"
                           f"Prompts: {self.temp_prompts_geojson}\n"
                           f"Predictions: {self.temp_predictions_geojson}")

                # Create empty GeoJSON files
                logger.info("Creating empty GeoJSON files")
                self.create_empty_geojson(self.temp_prompts_geojson)
                self.create_empty_geojson(self.temp_predictions_geojson)

                # Create and add layers
                logger.info("Creating vector layers")
                self.prompts_layer = self.create_vector_layer(
                    self.temp_prompts_geojson,
                    "Drawing Prompts",
                    "prompt"
                )
                self.predictions_layer = self.create_vector_layer(
                    self.temp_predictions_geojson,
                    "SAM Predictions",
                    "prediction"
                )

                # Initialize drawing tools
                logger.info("Initializing drawing tools")
                if not hasattr(self, 'map_tool'):
                    logger.error("map_tool not initialized!")
                    raise Exception("Drawing tool not properly initialized")
                    
                if not hasattr(self, 'canvas'):
                    logger.error("canvas not initialized!")
                    raise Exception("Canvas not properly initialized")
                    
                self.canvas.setMapTool(self.map_tool)
                
                # Initialize rubber bands if not already done
                if not hasattr(self, 'rubber_band'):
                    logger.info("Creating new rubber band")
                    self.rubber_band = QgsRubberBand(self.canvas, QgsWkbTypes.PointGeometry)
                    self.rubber_band.setColor(QColor(255, 0, 0))
                    self.rubber_band.setWidth(2)
                
                if not hasattr(self, 'temp_rubber_band'):
                    logger.info("Creating temporary rubber band")
                    self.temp_rubber_band = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
                    self.temp_rubber_band.setColor(QColor(255, 0, 0, 50))
                    self.temp_rubber_band.setWidth(2)

                self.draw_button.setText("Stop Drawing")
                
                logger.info("Drawing session started successfully")
                
            else:
                logger.info("Stopping drawing session")
                self.canvas.unsetMapTool(self.map_tool)
                self.draw_button.setText("Start Drawing")
                logger.info("Drawing session stopped")

        except Exception as e:
            logger.error(f"Failed to toggle drawing: {str(e)}")
            logger.exception("Full traceback:")
            QMessageBox.critical(None, "Error", f"Failed to start drawing: {str(e)}")
            self.draw_button.setChecked(False)

    def cleanup_previous_session(self):
        """Clean up temporary files and layers from previous session"""
        try:
            # Remove existing layers
            if self.prompts_layer and self.prompts_layer.isValid():
                QgsProject.instance().removeMapLayer(self.prompts_layer.id())
            if self.predictions_layer and self.predictions_layer.isValid():
                QgsProject.instance().removeMapLayer(self.predictions_layer.id())

            # Remove existing temporary files
            if self.temp_prompts_geojson and os.path.exists(self.temp_prompts_geojson):
                os.remove(self.temp_prompts_geojson)
            if self.temp_predictions_geojson and os.path.exists(self.temp_predictions_geojson):
                os.remove(self.temp_predictions_geojson)

            # Reset feature count
            self.feature_count = 0
            
            # Clear any existing rubber bands
            if hasattr(self, 'rubber_band') and self.rubber_band:
                self.rubber_band.reset()
            if hasattr(self, 'temp_rubber_band') and self.temp_rubber_band:
                self.temp_rubber_band.reset()
            
            # Reset start point for box drawing
            self.start_point = None

        except Exception as e:
            logger.error(f"Error cleaning up previous session: {str(e)}")
            raise

    def stop_drawing(self):
        """Stop drawing mode"""
        self.is_drawing = False
        self.draw_button.setText("Start Drawing")
        self.draw_type_combo.setEnabled(True)
        self.iface.mapCanvas().unsetMapTool(self.draw_tool)
        if self.temp_rubber_band:
            self.temp_rubber_band.reset()
        self.start_point = None

    def handle_draw_click(self, point, button):
        """Handle canvas clicks for drawing"""
        try:
            if not self.draw_button.isChecked() or button != Qt.LeftButton:
                return

            draw_type = self.draw_type_combo.currentText()
            
            # Get the raster layer
            raster_layer = None
            if self.source_combo.currentText() == "File":
                # Find the raster layer by name "Selected Image"
                for layer in QgsProject.instance().mapLayers().values():
                    if isinstance(layer, QgsRasterLayer) and layer.name() == "Selected Image":
                        raster_layer = layer
                        break
            else:
                raster_layer = self.layer_combo.currentData()

            if not raster_layer:
                raise ValueError("No raster layer found")

            # Get raster dimensions and extent
            extent = raster_layer.extent()
            width = raster_layer.width()
            height = raster_layer.height()

            if draw_type == "Point":
                # Reset rubber band for each new point to prevent line creation
                self.rubber_band.reset(QgsWkbTypes.PointGeometry)
                self.rubber_band.addPoint(point)
                
                # Calculate pixel coordinates
                px = int((point.x() - extent.xMinimum()) * width / extent.width())
                py = int((extent.yMaximum() - point.y()) * height / extent.height())
                
                # Ensure coordinates are within image bounds
                px = max(0, min(px, width - 1))
                py = max(0, min(py, height - 1))
                
                # Show coordinates in message bar
                self.iface.messageBar().pushMessage(
                    "Point Info", 
                    f"Map coordinates: ({point.x():.2f}, {point.y():.2f})\n"
                    f"Pixel coordinates sent to server: ({px}, {py})",
                    level=Qgis.Info,
                    duration=3
                )
                
                # Prepare point prompt
                prompt = [{
                    'type': 'Point',
                    'data': {
                        "points": [[px, py]],
                        "labels": [1]
                    }
                }]
                
                # Get prediction immediately
                self.get_prediction(prompt)
                
            elif draw_type == "Box":
                if not self.start_point:
                    # Start drawing box
                    self.start_point = point
                    self.temp_rubber_band.reset(QgsWkbTypes.PolygonGeometry)
                    self.temp_rubber_band.addPoint(point)
                    
                    # Show start point coordinates
                    self.iface.messageBar().pushMessage(
                        "Box Start Point", 
                        f"Start point - Map: ({point.x():.2f}, {point.y():.2f})",
                        level=Qgis.Info,
                        duration=3
                    )
                else:
                    # Calculate pixel coordinates for start point
                    start_px = int((self.start_point.x() - extent.xMinimum()) * width / extent.width())
                    start_py = int((extent.yMaximum() - self.start_point.y()) * height / extent.height())
                    
                    # Calculate pixel coordinates for end point
                    end_px = int((point.x() - extent.xMinimum()) * width / extent.width())
                    end_py = int((extent.yMaximum() - point.y()) * height / extent.height())
                    
                    # Ensure coordinates are within image bounds
                    start_px = max(0, min(start_px, width - 1))
                    start_py = max(0, min(start_py, height - 1))
                    end_px = max(0, min(end_px, width - 1))
                    end_py = max(0, min(end_py, height - 1))
                    
                    # Show box coordinates
                    self.iface.messageBar().pushMessage(
                        "Box Info", 
                        f"Box coordinates sent to server: [{min(start_px, end_px)}, {min(start_py, end_py)}, "
                        f"{max(start_px, end_px)}, {max(start_py, end_py)}]",
                        level=Qgis.Info,
                        duration=3
                    )
                    
                    # Create box geometry for display
                    box_geom = self.create_box_geometry(self.start_point, point)
                    self.rubber_band.addGeometry(box_geom)
                    
                    # Prepare box prompt
                    prompt = [{
                        'type': 'Box',
                        'data': {
                            "boxes": [[
                                min(start_px, end_px),
                                min(start_py, end_py),
                                max(start_px, end_px),
                                max(start_py, end_py)
                            ]]
                        }
                    }]
                    
                    # Get prediction immediately
                    self.get_prediction(prompt)
                    
                    # Reset for next box
                    self.temp_rubber_band.reset()
                    self.start_point = None

        except Exception as e:
            logger.error(f"Error handling draw click: {str(e)}")
            logger.exception("Full traceback:")
            QMessageBox.critical(None, "Error", f"Failed to handle drawing: {str(e)}")

    def handle_mouse_move(self, event):
        """Handle mouse movement for box drawing"""
        if not self.start_point:
            return

        point = self.draw_tool.toMapCoordinates(event.pos())
        self.temp_rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        self.temp_rubber_band.addGeometry(self.create_box_geometry(self.start_point, point))

    def create_box_geometry(self, start, end):
        """Create a box geometry from two points"""
        points = [
            QgsPointXY(start.x(), start.y()),
            QgsPointXY(end.x(), start.y()),
            QgsPointXY(end.x(), end.y()),
            QgsPointXY(start.x(), end.y()),
            QgsPointXY(start.x(), start.y())
        ]
        return QgsGeometry.fromPolygonXY([points])

    def finish_drawing(self):
        """Finish drawing and keep features in memory"""
        try:
            self.stop_drawing()
            self.finish_button.setEnabled(False)
            self.save_button.setEnabled(True)
            
            # Display feature count in message bar
            self.iface.messageBar().pushMessage(
                "Success", 
                f"Drawing completed. {len(self.drawn_features)} features drawn.", 
                level=Qgis.Success
            )

        except Exception as e:
            logger.error(f"Error in finish_drawing: {str(e)}")
            logger.exception("Full traceback:")
            QMessageBox.critical(None, "Error", f"Failed to finish drawing: {str(e)}")

    def save_features(self):
        """Save drawn features to a permanent file"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self.dock_widget,
                "Save Features As",
                "",
                "GeoJSON (*.geojson);;All Files (*.*)"
            )
            
            if not file_path:  # User cancelled
                return

            # Ensure file has .geojson extension
            if not file_path.endswith('.geojson'):
                file_path += '.geojson'

            # Copy temporary file to selected location
            shutil.copy2(self.temp_geojson_path, file_path)

            self.iface.messageBar().pushMessage(
                "Success", 
                f"Features saved to: {file_path}", 
                level=Qgis.Success,
                duration=5
            )

        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            logger.exception("Full traceback:")
            QMessageBox.critical(None, "Error", f"Failed to save features: {str(e)}")

    def get_prediction(self, prompts):
        """Get prediction from SAM server and add to predictions layer"""
        try:
            # First check if server is running
            if not self.server_running:
                raise ConnectionError("Server is not running. Please ensure Docker is started and the server is running.")

            # Get the image path
            image_path = self.image_path.text()
            if not image_path:
                raise ValueError("No image selected")
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")

            # Validate prompts format
            if not isinstance(prompts, list) or not prompts:
                raise ValueError("Invalid prompts format")
            
            # Get embedding settings
            embedding_path = None
            save_embeddings = False

            if self.load_embedding_radio.isChecked():
                embedding_path = self.embedding_path_edit.text().strip()
                if not embedding_path:
                    raise ValueError("Please select an embedding file to load")
                if not os.path.exists(embedding_path):
                    raise ValueError(f"Embedding file not found: {embedding_path}")
                save_embeddings = False
            elif self.save_embedding_radio.isChecked():
                embedding_path = self.embedding_path_edit.text().strip()
                if not embedding_path:
                    raise ValueError("Please specify a path to save the embedding")
                embedding_dir = os.path.dirname(embedding_path)
                if not os.path.exists(embedding_dir):
                    raise ValueError(f"Directory does not exist: {embedding_dir}")
                if not os.access(embedding_dir, os.W_OK):
                    raise ValueError(f"No write permission for directory: {embedding_dir}")
                save_embeddings = True

            # Prepare request payload
            payload = {
                "image_path": image_path,
                "embedding_path": embedding_path,
                "prompts": prompts,
                "save_embeddings": save_embeddings
            }

            # Show payload in message bar
            formatted_payload = (
                f"Sending to server:\n"
                f"- Host image path: {image_path}\n"
                f"- Prompt type: {prompts[0]['type']}\n"
            )
            
            if prompts[0]['type'] == 'Point':
                points = prompts[0]['data']['points']
                formatted_payload += f"- Points: {points}\n"
                formatted_payload += f"- Labels: {prompts[0]['data']['labels']}"
            elif prompts[0]['type'] == 'Box':
                boxes = prompts[0]['data']['boxes']
                formatted_payload += f"- Box coordinates: {boxes}"

            self.iface.messageBar().pushMessage(
                "Server Request", 
                formatted_payload,
                level=Qgis.Info,
                duration=5
            )

            # Log detailed request information
            logger.debug("=== Request Details ===")
            logger.debug(f"Server URL: {self.server_url}/sam-predict")
            logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")

            # Send request to SAM server
            try:
                response = requests.post(
                    f"{self.server_url}/sam-predict",
                    json=payload,
                    timeout=60000  # TODO: it is very slow at the moment
                )
                
                # Log the response details
                logger.debug("=== Response Details ===")
                logger.debug(f"Status code: {response.status_code}")
                logger.debug(f"Response headers: {dict(response.headers)}")
                
                try:
                    response_json = response.json()
                    logger.debug(f"Response body: {json.dumps(response_json, indent=2)}")
                except Exception as e:
                    logger.debug(f"Raw response: {response.text}")

                if response.status_code == 400:
                    error_msg = "Bad request. Server rejected the input format."
                    if response_json and 'message' in response_json:
                        error_msg = f"Server error: {response_json['message']}"
                    raise Exception(error_msg)
                
                response.raise_for_status()  # Raise exception for other error status codes
                
                result = response_json
                if 'features' not in result:
                    raise ValueError("Response missing 'features' field")

                # Process features and add to layer
                if result['features']:
                    self.add_predictions_to_layer(result['features'])
                    self.iface.messageBar().pushMessage(
                        "Success", 
                        f"Added {len(result['features'])} prediction(s) to layer",
                        level=Qgis.Success,
                        duration=2
                    )
                else:
                    self.iface.messageBar().pushMessage(
                        "Warning", 
                        "No predictions were received",
                        level=Qgis.Warning,
                        duration=2
                    )

            except Exception as e:
                logger.error(f"Error getting prediction: {str(e)}")
                logger.exception("Full traceback:")
                QMessageBox.critical(None, "Error", f"Failed to get prediction: {str(e)}")

        except Exception as e:
            logger.error(f"Error getting prediction: {str(e)}")
            logger.exception("Full traceback:")
            QMessageBox.critical(None, "Error", f"Failed to get prediction: {str(e)}")

    def add_predictions_to_layer(self, features):
        """Add multiple GeoJSON features to the predictions layer at once"""
        try:
            logger.debug(f"Adding {len(features)} features to layer")
            
            if not self.predictions_layer:
                logger.error("Predictions layer not initialized")
                raise ValueError("Predictions layer not initialized")

            # Create list to hold all QGIS features
            qgis_features = []
            fields = self.predictions_layer.fields()

            # Convert all features
            for feature in features:
                qgis_feature = QgsFeature()
                
                # TODO: fix this part
                # Convert geometry
                geom_json = feature.get('geometry')
                if geom_json:
                    geometry = QgsGeometry.fromJson(json.dumps(geom_json))
                    if geometry:
                        qgis_feature.setGeometry(geometry)
                
                # Set attributes from properties
                properties = feature.get('properties', {})
                attributes = []
                for field in fields:
                    field_name = field.name()
                    field_value = properties.get(field_name, None)
                    attributes.append(field_value)
                
                qgis_feature.setAttributes(attributes)
                qgis_features.append(qgis_feature)

            # Add all features at once
            if not self.predictions_layer.dataProvider().addFeatures(qgis_features):
                logger.error("Failed to add features to layer")
                raise ValueError("Failed to add features to layer")
            
            self.predictions_layer.updateExtents()
            self.predictions_layer.triggerRepaint()
            
            logger.debug("Successfully added all features to layer")

        except Exception as e:
            logger.error(f"Error adding predictions to layer: {str(e)}")
            logger.exception("Full traceback:")
            raise

    def toggle_prediction_mode(self, state):
        """Handle prediction mode change"""
        self.real_time_prediction = bool(state)
        self.predict_button.setEnabled(not self.real_time_prediction)

    def check_docker_compose(self):
        """Check if docker-compose.yml exists"""
        compose_path = os.path.join(self.plugin_dir, 'docker-compose.yml')
        if not os.path.exists(compose_path):
            QMessageBox.critical(None, "Error", 
                f"docker-compose.yml not found at: {compose_path}\n"
                "Please ensure the file exists in the plugin directory.")
            return False
        return True

    def cleanup_tmp_files(self):
        """Clean up temporary files"""
        try:
            tmp_compose_path = '/tmp/docker-compose.yml'
            if os.path.exists(tmp_compose_path):
                os.remove(tmp_compose_path)
        except Exception as e:
            logger.error(f"Failed to cleanup temporary files: {str(e)}")

    def count_docker_steps(self):
        """Count the total number of steps in docker-compose and Dockerfile"""
        try:
            # Check if image exists first
            if self.check_docker_image():
                # Fewer steps if just starting existing container
                self.total_steps = 3  # pull if needed, create container, start container
                self.progress_bar.setMaximum(self.total_steps)
                return

            compose_path = os.path.join(self.plugin_dir, 'docker-compose.yml')
            dockerfile_path = os.path.join(self.plugin_dir, 'Dockerfile')
            
            steps = 0
            
            # Count steps in Dockerfile
            if os.path.exists(dockerfile_path):
                with open(dockerfile_path, 'r') as f:
                    content = f.read()
                    for line in content.split('\n'):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        if line.startswith('RUN pip install'):
                            # Count requirements.txt entries if referenced
                            if 'requirements.txt' in line:
                                req_path = os.path.join(self.plugin_dir, 'requirements.txt')
                                if os.path.exists(req_path):
                                    with open(req_path, 'r') as req_file:
                                        # Count non-empty, non-comment lines in requirements.txt
                                        req_steps = sum(1 for l in req_file if l.strip() and not l.strip().startswith('#'))
                                        steps += req_steps
                            else:
                                # Count individual pip install packages
                                packages = line.count('>=') + line.count('==') + line.count('@')
                                steps += max(packages, 1)
                        elif any(line.startswith(cmd) for cmd in [
                            'FROM', 'COPY', 'ADD', 'RUN', 'ENV', 'WORKDIR',
                            'EXPOSE', 'VOLUME', 'CMD', 'ENTRYPOINT'
                        ]):
                            steps += 1
            
            # Add steps for Docker Compose operations
            steps += 5  # network creation, volume creation, image build, container creation, container start
            
            self.total_steps = max(steps, 1)  # Ensure at least 1 step
            self.progress_bar.setMaximum(self.total_steps)
            logger.debug(f"Total build steps: {self.total_steps}")
            
        except Exception as e:
            logger.error(f"Error counting docker steps: {str(e)}")
            self.total_steps = 0

    def _process_docker_output(self, output):
        """Process Docker output and update progress"""
        progress_indicators = {
            'Pulling': 1,
            'Pull complete': 1,
            'Downloading': 1,
            'Download complete': 1,
            'Extracting': 1,
            'Extract complete': 1,
            'Building': 1,
            'Step': 1,
            'Running': 1,
            'Creating network': 1,
            'Creating volume': 1,
            'Creating container': 1,
            'Starting container': 1,
            'Collecting': 1,  # For pip install progress
            'Installing collected packages': 1,
            'Successfully installed': 1,
            'Requirement already satisfied': 1
        }

        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if line:
                # Special handling for pip install progress
                if 'pip install' in line:
                    self.progress_status.setText("Installing dependencies...")
                    self.progress_status.show()
                elif 'Successfully built' in line:
                    self.progress_bar.setValue(self.total_steps)
                    self.progress_status.setText("Build completed successfully")
                    return

                for indicator in progress_indicators:
                    if indicator in line:
                        self.current_step += 1
                        progress = min(self.current_step, self.total_steps)
                        self.progress_bar.setValue(progress)
                        self.progress_status.setText(line)
                        self.progress_status.show()
                        logger.debug(f"Docker progress: {line}")
                        break

    def check_server_status(self):
        """Check if the server is running by pinging it"""
        try:
            response = requests.get(f"http://0.0.0.0:{self.server_port}/v1/easyearth/ping", timeout=2)
            if response.status_code == 200:
                self.server_status.setText("Running")
                self.server_status.setStyleSheet("color: green;")
                self.server_running = True
            else:
                self.server_status.setText("Error")
                self.server_status.setStyleSheet("color: red;")
                self.server_running = False
        except requests.exceptions.RequestException:
            self.server_status.setText("Not Running")
            self.server_status.setStyleSheet("color: red;")
            self.server_running = False

    def cleanup_docker(self):
        """Clean up Docker resources when unloading plugin"""
        try:
            if self.sudo_password:
                compose_path = os.path.join(self.plugin_dir, 'docker-compose.yml')
                cmd = f'echo "{self.sudo_password}" | sudo -S docker-compose -p {self.project_name} -f "{compose_path}" down'
                subprocess.run(['bash', '-c', cmd], check=True)
        except Exception as e:
            logger.error(f"Error cleaning up Docker: {str(e)}")

    def on_embedding_option_changed(self, button):
        """Handle embedding option changes"""
        try:
            logger.debug(f"Embedding option changed to: {button.text()}")
            enable_path = button != self.no_embedding_radio
            self.embedding_path_edit.setEnabled(enable_path)
            self.embedding_browse_btn.setEnabled(enable_path)
            
            if not enable_path:
                self.embedding_path_edit.clear()
                logger.debug("Cleared embedding path")
            
            logger.debug(f"Path input enabled: {enable_path}")
        except Exception as e:
            logger.error(f"Error in embedding option change: {str(e)}")

    def browse_embedding(self):
        """Browse for embedding file"""
        try:
            if self.load_embedding_radio.isChecked():
                # Browse for existing file
                file_path, _ = QFileDialog.getOpenFileName(
                    None,
                    "Select Embedding File",
                    "",
                    "Embedding Files (*.pt);;All Files (*.*)"
                )
                logger.debug(f"Selected existing embedding file: {file_path}")
            else:
                # Browse for save location
                file_path, _ = QFileDialog.getSaveFileName(
                    None,
                    "Save Embedding As",
                    "",
                    "Embedding Files (*.pt);;All Files (*.*)"
                )
                logger.debug(f"Selected save location for embedding: {file_path}")

            if file_path:
                self.embedding_path_edit.setText(file_path)
                logger.debug(f"Set embedding path to: {file_path}")

        except Exception as e:
            logger.error(f"Error browsing embedding: {str(e)}")
            QMessageBox.critical(None, "Error", f"Failed to browse embedding: {str(e)}")

    def create_empty_geojson(self, file_path):
        """Create an empty GeoJSON file with project CRS"""
        try:
            # Get project CRS
            project_crs = QgsProject.instance().crs()
            
            # Create GeoJSON structure with CRS information
            empty_geojson = {
                "type": "FeatureCollection",
                "name": os.path.basename(file_path),
                "crs": {
                    "type": "name",
                    "properties": {
                        "name": project_crs.authid()
                    }
                },
                "features": []
            }
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(empty_geojson, f)
            
        except Exception as e:
            logger.error(f"Error creating empty GeoJSON: {str(e)}")
            raise

    def add_feature_to_geojson(self, geojson_path, geometry, feature_type, layer):
        """Add a feature to GeoJSON file and layer with correct CRS"""
        try:
            # Create feature
            feature = QgsFeature()
            
            # Ensure geometry is in project CRS
            project_crs = QgsProject.instance().crs()
            if geometry.transform(QgsCoordinateTransform(
                layer.crs(), 
                project_crs,
                QgsProject.instance()
            )) != 0:
                raise Exception("Failed to transform geometry to project CRS")
            
            feature.setGeometry(geometry)
            
            # Set attributes
            self.feature_count += 1
            feature.setAttributes([
                self.feature_count,
                feature_type
            ])
            
            # Add to layer
            layer.dataProvider().addFeatures([feature])
            layer.updateExtents()
            layer.triggerRepaint()
            
            return feature
            
        except Exception as e:
            logger.error(f"Error adding feature to GeoJSON: {str(e)}")
            raise

    def style_drawn_features(self):
        """Apply styling to the drawn features layer"""
        if not self.drawn_layer:
            return

        # Create symbol for points
        point_symbol = QgsMarkerSymbol.createSimple({
            'name': 'circle',
            'color': 'red',
            'size': '4',
        })

        # Create symbol for polygons (boxes)
        polygon_symbol = QgsFillSymbol.createSimple({
            'color': 'rgba(255, 0, 0, 50)',
            'outline_color': 'red',
            'outline_width': '1',
        })

        # Create categories
        categories = [
            QgsRendererCategory('Point', point_symbol, 'Point'),
            QgsRendererCategory('Box', polygon_symbol, 'Box')
        ]

        # Create and apply the renderer
        renderer = QgsCategorizedSymbolRenderer('type', categories)
        self.drawn_layer.setRenderer(renderer)

        # Refresh the layer
        self.drawn_layer.triggerRepaint()

    def on_draw_type_changed(self, draw_type):
        """Handle draw type change"""
        try:
            logger.debug(f"Draw type changed to: {draw_type}")
            if self.draw_button.isChecked():
                # If currently drawing, restart with new type
                self.draw_button.setChecked(False)
                self.toggle_drawing(True)
        except Exception as e:
            logger.error(f"Error in draw type change: {str(e)}")