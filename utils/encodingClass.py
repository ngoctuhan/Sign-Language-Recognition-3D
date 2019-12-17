import io

def load_doc(filename):
	# open the file as read only
	file = io.open(filename, mode="r", encoding="utf-8")
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
def loadLabel(filename):

    text = load_doc(filename)
    CLASSES = {}
    for line in text.split('\n'):
        lineSplit = line.split(' : ')
        
        code = int(lineSplit[0])
        name = lineSplit[1]
        CLASSES[name] = code
        
    return CLASSES

# CLASSES = loadLabel()

def loadLabelN2W(filename):
    text = load_doc(filename)
    CLASSES = {}
    for line in text.split('\n'):
        lineSplit = line.split(' : ')
        
        code = int(lineSplit[0])
        name = lineSplit[1]
        CLASSES[code] = name
        
    return CLASSES


