# add our app to the system path
import sys
sys.path.insert(0, "/var/www/html/Lashinbang/")

# import the application and away we go...
from cbir_app import app as application