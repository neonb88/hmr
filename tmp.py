from demo import *
sys.argv=['demo.py', '--img_path', '/home/n/Pictures/jonah_hill.jpeg']
config(sys.argv) 
config.load_path = src.config.PRETRAINED_MODEL
config.batch_size = 1
shapedirs=main3(config.img_path, config.json_path) 
