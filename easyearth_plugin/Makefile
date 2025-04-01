PLUGIN_NAME = EasyEarth

deploy: clean compile
	mkdir -p $(HOME)/.local/share/QGIS/QGIS3/profiles/default/python/plugins/$(PLUGIN_NAME)
	cp -r * $(HOME)/.local/share/QGIS/QGIS3/profiles/default/python/plugins/$(PLUGIN_NAME)

compile:
	pyuic5 -o ui/sam_dialog.py ui/sam_dialog.ui
	pyrcc5 -o resources_rc.py resources.qrc

clean:
	rm -f ui/sam_dialog.py
	rm -f resources_rc.py
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete