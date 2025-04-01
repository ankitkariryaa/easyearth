import unittest
from qgis.PyQt.QtCore import Qt
from qgis.testing import start_app, unittest
from qgis.core import QgsApplication

class TestSAMPlugin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        start_app()

    def test_plugin_init(self):
        """Test plugin initialization"""
        from ..plugin import SAMPlugin
        plugin = SAMPlugin(QgsApplication.instance())
        self.assertIsNotNone(plugin)

if __name__ == "__main__":
    unittest.main()