from qgis.PyQt.QtWidgets import (QAction, QDockWidget, QPushButton, QVBoxLayout, 
                                QWidget, QMessageBox, QLabel, QHBoxLayout,
                                QLineEdit, QFileDialog, QComboBox, QGroupBox, QGridLayout)
from qgis.PyQt.QtCore import Qt, QByteArray, QBuffer, QIODevice, QProcess, QTimer
from qgis.PyQt.QtGui import QIcon
from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry, QgsPolygon, 
                      QgsPointXY, QgsField, QgsProject, QgsPoint, QgsLineString,
                      QgsWkbTypes, QgsRasterLayer, Qgis)
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

class EasyEarthPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.actions = []
        self.menu = 'EasyEarth'
        self.toolbar = self.iface.addToolBar(u'EasyEarth')
        self.toolbar.setObjectName(u'EasyEarth')
        
        # Initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        
        # Initialize map tools and data
        self.canvas = iface.mapCanvas()
        self.point_tool = None
        self.points = []
        self.rubber_bands = []
        self.docker_process = None
        self.server_process = None
        self.server_port = 3781  # Default port
        self.server_url = f"http://localhost:{self.server_port}/v1/easyearth"
        self.current_image_path = None
        self.current_embedding_path = None
        self.docker_running = False
        self.server_running = False
        self.action = None
        self.dock_widget = None
        self.rubber_band = None
        self.is_selecting_points = False

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
        
        # Create the plugin button
        self.action = QAction('EasyEarth Plugin', self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)

        # Create dock widget
        self.dock_widget = QDockWidget('EasyEarth Plugin', self.iface.mainWindow())
        self.dock_widget.setObjectName('EasyEarthPluginDock')
        
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Service Control Group
        service_group = QGroupBox("Service Control")
        service_layout = QVBoxLayout()  # Changed to QVBoxLayout for simpler nesting

        # Docker controls
        docker_layout = QHBoxLayout()
        docker_label = QLabel("Docker Status:")
        self.docker_status = QLabel("Stopped")
        self.docker_button = QPushButton("Start Docker")
        self.docker_button.clicked.connect(self.toggle_docker)
        docker_layout.addWidget(docker_label)
        docker_layout.addWidget(self.docker_status)
        docker_layout.addWidget(self.docker_button)
        service_layout.addLayout(docker_layout)

        # Server controls
        server_layout = QHBoxLayout()
        server_label = QLabel("Server Status:")
        self.server_status = QLabel("Stopped")
        self.server_button = QPushButton("Start Server")
        self.server_button.clicked.connect(self.toggle_server)
        server_layout.addWidget(server_label)
        server_layout.addWidget(self.server_status)
        server_layout.addWidget(self.server_button)
        service_layout.addLayout(server_layout)

        service_group.setLayout(service_layout)
        main_layout.addWidget(service_group)

        # Image Source Group
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
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_image)
        file_layout.addWidget(self.image_path)
        file_layout.addWidget(self.browse_button)
        image_layout.addLayout(file_layout)
        
        # Layer selection
        self.layer_combo = QComboBox()
        self.layer_combo.hide()
        image_layout.addWidget(self.layer_combo)
        
        # Load button
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        image_layout.addWidget(self.load_button)
        
        image_group.setLayout(image_layout)
        main_layout.addWidget(image_group)

        # Point Selection Group
        point_group = QGroupBox("Point Selection")
        point_layout = QVBoxLayout()
        self.point_button = QPushButton("Add Point")
        self.point_button.clicked.connect(self.toggle_point_selection)
        point_layout.addWidget(self.point_button)
        point_group.setLayout(point_layout)
        main_layout.addWidget(point_group)

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
        is_file_source = text == "File"
        self.image_path.setVisible(is_file_source)
        self.browse_button.setVisible(is_file_source)
        self.layer_combo.setVisible(not is_file_source)
        
        if not is_file_source:
            self.update_layer_combo()

    def browse_image(self):
        """Open file dialog for image selection"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.dock_widget,
            "Select Image File",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*.*)"
        )
        if file_path:
            self.image_path.setText(file_path)

    def browse_embedding(self):
        """Open file dialog for embedding selection"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.dock_widget,
            "Select Embedding File",
            "",
            "Embedding Files (*.pt *.pth);;All Files (*.*)"
        )
        if file_path:
            self.embedding_path_edit.setText(file_path)

    def load_image(self):
        """Load the selected image"""
        try:
            if self.source_combo.currentText() == "File":
                image_path = self.image_path.text()
                if not image_path:
                    QMessageBox.warning(None, "Warning", "Please select an image file")
                    return
                    
                # Load image as raster layer
                layer_name = os.path.basename(image_path)
                raster_layer = QgsRasterLayer(image_path, layer_name)
                
            else:
                # Get selected layer
                current_layer = self.layer_combo.currentData()
                if not current_layer:
                    QMessageBox.warning(None, "Warning", "Please select a layer")
                    return
                    
                image_path = current_layer.source()
                raster_layer = current_layer

            if not raster_layer.isValid():
                QMessageBox.warning(None, "Error", "Failed to load image layer")
                return

            # Add layer to map if it's not already there
            if self.source_combo.currentText() == "File":
                QgsProject.instance().addMapLayer(raster_layer)

            # Store the image path
            self.current_image_path = image_path
            self.current_embedding_path = self.embedding_path_edit.text()

            # Enable point capture
            self.point_button.setEnabled(True)
            
            # Zoom to the layer
            self.iface.setActiveLayer(raster_layer)
            self.canvas.setExtent(raster_layer.extent())
            self.canvas.refresh()

            QMessageBox.information(None, "Success", "Image loaded successfully")
            
        except Exception as e:
            QMessageBox.warning(None, "Error", f"Failed to load image: {str(e)}")

    def start_point_capture(self):
        """Start capturing points on the map"""
        self.point_tool = QgsMapToolEmitPoint(self.canvas)
        self.point_tool.canvasClicked.connect(self.capture_point)
        self.canvas.setMapTool(self.point_tool)
        self.point_button.setEnabled(False)
        
    def capture_point(self, point, button):
        """Capture clicked point"""
        try:
            x, y = point.x(), point.y()
            
            # Add point to list
            self.points.append({
                "type": "Point",
                "data": {
                    "x": x,
                    "y": y
                }
            })
            
            # Add visual marker
            self.add_point_marker(x, y)
            
            # Update point counter
            self.point_counter.setText(f"Points: {len(self.points)}")
            
            # Re-enable add point button
            self.point_button.setEnabled(True)
            
        except Exception as e:
            QMessageBox.warning(None, "Error", f"Failed to capture point: {str(e)}")
            
    def add_point_marker(self, x, y):
        """Add a visual marker for captured points"""
        try:
            # Create rubber band with correct geometry type
            rubber_band = QgsRubberBand(self.canvas, QgsWkbTypes.PointGeometry)  # Use point geometry type
            rubber_band.setColor(Qt.red)
            rubber_band.setWidth(3)
            rubber_band.setIcon(QgsRubberBand.ICON_CIRCLE)  # Set point style to circle
            rubber_band.setIconSize(10)  # Set size of the point marker
            rubber_band.addPoint(QgsPointXY(x, y))
            self.rubber_bands.append(rubber_band)
        except Exception as e:
            QMessageBox.warning(None, "Error", f"Failed to add point marker: {str(e)}")
            
    def send_to_server(self):
        """Send the image and points to the easyearth for prediction"""
        try:
            # Check if easyearth is running
            if not self.server_process or self.server_process.state() != QProcess.Running:
                QMessageBox.warning(None, "Warning", "Server is not running")
                return

            if not self.current_image_path:
                QMessageBox.warning(None, "Warning", "Please load an image first")
                return

            if not self.points:
                QMessageBox.warning(None, "Warning", "No points selected")
                return

            # Prepare request data
            data = {
                "image_path": self.current_image_path,
                "embedding_path": self.current_embedding_path if self.current_embedding_path else None,
                "prompts": self.points
            }
            
            # Send request to easyearth
            self.predict_button.setEnabled(False)
            self.predict_button.setText("Processing...")
            
            response = requests.post(f"{self.server_url}/sam-predict", json=data)
            
            if response.status_code != 200:
                QMessageBox.warning(None, "Error", f"Server returned: {response.text}")
                return
                
            result = response.json()
            
            # Process predictions
            if 'predictions' in result:
                for i, pred in enumerate(result['predictions']):
                    if pred['type'] == 'Polygons':
                        self.create_vector_layer_from_prediction(
                            pred['data'],
                            f"SAM_prediction_{i+1}"
                        )
            
        except Exception as e:
            QMessageBox.warning(None, "Error", f"Failed to process image: {str(e)}")
        finally:
            self.predict_button.setEnabled(True)
            self.predict_button.setText("Predict")
            
    def create_vector_layer_from_prediction(self, pred_data, layer_name):
        """Convert prediction data to vector layer"""
        try:
            # Create vector layer
            layer = QgsVectorLayer("Polygon", layer_name, "memory")
            provider = layer.dataProvider()
            
            # Add fields
            provider.addAttributes([
                QgsField("confidence", QVariant.Double),
                QgsField("class", QVariant.String)
            ])
            layer.updateFields()
            
            # Create features from prediction polygons
            for polygon_data in pred_data.get('polygons', []):
                feature = QgsFeature()
                
                # Convert coordinates to QgsPolygon
                points = [QgsPointXY(p[0], p[1]) for p in polygon_data['coordinates']]
                ring = QgsLineString([QgsPoint(p.x(), p.y()) for p in points])
                polygon = QgsPolygon()
                polygon.setExteriorRing(ring)
                feature.setGeometry(QgsGeometry.fromPolygon(polygon))
                
                # Set attributes
                feature.setAttributes([
                    polygon_data.get('confidence', 0.0),
                    polygon_data.get('class', '')
                ])
                
                provider.addFeature(feature)
            
            # Add layer to map
            QgsProject.instance().addMapLayer(layer)
            
            # Style the layer
            self.style_prediction_layer(layer)
            
        except Exception as e:
            QMessageBox.warning(None, "Error", f"Failed to create vector layer: {str(e)}")
    
    def style_prediction_layer(self, layer):
        """Apply styling to the prediction layer"""
        # Set layer symbology
        symbol = layer.renderer().symbol()
        symbol.setColor(Qt.red)
        symbol.setOpacity(0.3)
        
        # Refresh the layer
        layer.triggerRepaint()
        self.iface.layerTreeView().refreshLayerSymbology(layer.id())
    
    def clear_points(self):
        """Clear all captured points"""
        self.points = []
        # Clear rubber bands
        for rb in self.rubber_bands:
            self.canvas.scene().removeItem(rb)
        self.rubber_bands = []
        # Update counter
        self.point_counter.setText("Points: 0")
        # Re-enable add point button
        self.point_button.setEnabled(self.current_image_path is not None)

    def run(self):
        """Run method that loads and starts the plugin"""
        self.dock_widget.show()

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI"""
        # Stop easyearth if running
        if self.server_running:
            self.toggle_server()
            
        # Stop Docker if running
        if self.docker_running:
            self.toggle_docker()
            
        # Remove the plugin menu item and icon
        for action in self.actions:
            self.iface.removePluginMenu(u'SAM Plugin', action)
            self.iface.removeToolBarIcon(action)
        
        # Remove the toolbar
        del self.toolbar
        
        # Clear any rubber bands
        self.clear_points()
        
        # Remove the dock widget
        self.dock_widget.deleteLater()

    def toggle_docker(self):
        """Toggle Docker container state"""
        if not self.docker_running:
            try:
                # Assuming docker-compose.yml is in the same directory as the plugin
                compose_path = os.path.join(os.path.dirname(__file__), 'docker-compose.yml')
                
                # Start Docker containers
                process = QProcess()
                process.start('docker-compose', ['-f', compose_path, 'up', '-d'])
                
                # Wait for process to finish (30 seconds)
                if process.waitForFinished(30000):  # Remove 'timeout=' keyword
                    self.docker_running = True
                    self.docker_status.setText("Running")
                    self.docker_button.setText("Stop Docker")
                    QMessageBox.information(None, "Success", "Docker containers started successfully")
                else:
                    QMessageBox.warning(None, "Error", "Failed to start Docker containers")
            
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Error starting Docker: {str(e)}")
        else:
            try:
                # Stop Docker containers
                compose_path = os.path.join(os.path.dirname(__file__), 'docker-compose.yml')
                process = QProcess()
                process.start('docker-compose', ['-f', compose_path, 'down'])
                
                # Wait for process to finish (30 seconds)
                if process.waitForFinished(30000):  # Remove 'timeout=' keyword
                    self.docker_running = False
                    self.docker_status.setText("Stopped")
                    self.docker_button.setText("Start Docker")
                    QMessageBox.information(None, "Success", "Docker containers stopped successfully")
                else:
                    QMessageBox.warning(None, "Error", "Failed to stop Docker containers")
            
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Error stopping Docker: {str(e)}")

    def toggle_server(self):
        """Toggle easyearth state"""
        if not self.server_running:
            try:
                # Start the Flask easyearth
                server_script = os.path.join(self.plugin_dir, 'sam_server', 'app.py')
                
                self.server_process = QProcess()
                self.server_process.start('python3', [server_script])
                
                # Wait a bit for the easyearth to start
                time.sleep(2)
                
                # Try to connect to verify the easyearth is running
                try:
                    response = requests.get(f"{self.server_url}/ping")
                    if response.status_code == 200:
                        self.server_running = True
                        self.server_status.setText("Running")
                        self.server_button.setText("Stop Server")
                        QMessageBox.information(None, "Success", "Server started successfully")
                    else:
                        raise Exception("Server returned unexpected status code")
                except requests.exceptions.RequestException:
                    raise Exception("Could not connect to easyearth")
            
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Error starting easyearth: {str(e)}")
                if self.server_process:
                    self.server_process.kill()
                    self.server_process = None
        else:
            try:
                # Stop the easyearth
                if self.server_process:
                    self.server_process.kill()
                    self.server_process = None
                    self.server_running = False
                    self.server_status.setText("Stopped")
                    self.server_button.setText("Start Server")
                    QMessageBox.information(None, "Success", "Server stopped successfully")
            
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Error stopping easyearth: {str(e)}")

    def toggle_point_selection(self):
        """Toggle point selection mode"""
        if not self.is_selecting_points:
            # Start point selection
            self.is_selecting_points = True
            self.point_button.setText("Stop Point Selection")
            
            # Create map tool for point selection
            from qgis.gui import QgsMapToolEmitPoint
            self.point_tool = QgsMapToolEmitPoint(self.iface.mapCanvas())
            self.point_tool.canvasClicked.connect(self.add_point)
            
            # Create rubber band for visualization
            from qgis.core import QgsWkbTypes
            from qgis.gui import QgsRubberBand
            from qgis.PyQt.QtCore import Qt
            
            self.rubber_band = QgsRubberBand(self.iface.mapCanvas(), QgsWkbTypes.PointGeometry)
            self.rubber_band.setColor(Qt.red)
            self.rubber_band.setWidth(4)
            
            # Activate the point tool
            self.iface.mapCanvas().setMapTool(self.point_tool)
            
        else:
            # Stop point selection
            self.is_selecting_points = False
            self.point_button.setText("Add Point")
            
            # Deactivate the point tool
            self.iface.mapCanvas().unsetMapTool(self.point_tool)
            
            if self.rubber_band:
                # Clear the rubber band but keep the points
                self.rubber_band.reset(QgsWkbTypes.PointGeometry)

    def add_point(self, point, button):
        """Add a point to the collection"""
        if button == Qt.LeftButton and self.is_selecting_points:
            # Store the point coordinates
            self.points.append(point)
            
            # Add point to rubber band for visualization
            self.rubber_band.addPoint(point)
            
            # Optional: Show coordinates in message bar
            msg = f"Point added at {point.x():.4f}, {point.y():.4f}"
            self.iface.messageBar().pushMessage("Info", msg, level=Qgis.Info, duration=3)

    def get_prediction(self):
        """Get prediction from the SAM easyearth"""
        if not self.server_running:
            QMessageBox.warning(None, "Error", "Server is not running. Please start the easyearth first.")
            return

        if not self.points:
            QMessageBox.warning(None, "Error", "No points selected. Please add points first.")
            return

        try:
            # Get the current image path or layer
            image_path = None
            if self.source_combo.currentText() == "File":
                image_path = self.image_path.text()
            else:
                current_layer = self.layer_combo.currentLayer()
                if current_layer:
                    image_path = current_layer.source()

            if not image_path:
                QMessageBox.warning(None, "Error", "No image selected.")
                return

            # Prepare the points data
            points_data = [{"x": point.x(), "y": point.y()} for point in self.points]

            # Prepare the request data
            request_data = {
                "image_path": image_path,
                "points": points_data
            }

            # Send request to the easyearth
            response = requests.post(
                f"{self.server_url}/sam-predict",
                json=request_data
            )

            if response.status_code == 200:
                result = response.json()
                
                # Create a new vector layer for the prediction
                from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry, 
                                     QgsProject, QgsPolygon, QgsPointXY)
                
                # Create a memory layer for the prediction
                layer = QgsVectorLayer("Polygon", "SAM Prediction", "memory")
                provider = layer.dataProvider()
                
                # Add features to the layer
                features = []
                for polygon in result.get('polygons', []):
                    feature = QgsFeature()
                    points = [QgsPointXY(p['x'], p['y']) for p in polygon]
                    geometry = QgsGeometry.fromPolygonXY([points])
                    feature.setGeometry(geometry)
                    features.append(feature)
                
                provider.addFeatures(features)
                
                # Add the layer to the project
                QgsProject.instance().addMapLayer(layer)
                
                # Clear the points after successful prediction
                self.clear_points()
                
                QMessageBox.information(None, "Success", "Prediction completed successfully!")
            else:
                QMessageBox.warning(None, "Error", 
                                  f"Server returned error: {response.status_code}\n{response.text}")

        except requests.exceptions.RequestException as e:
            QMessageBox.critical(None, "Error", f"Failed to communicate with easyearth: {str(e)}")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"An error occurred: {str(e)}")