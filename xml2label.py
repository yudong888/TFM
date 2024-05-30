import xml.etree.ElementTree as ET 
import shutil, os


def saveText(filename): 
    tree = ET.parse('./' + dirs_list + '/' + filename)
    root = tree.getroot()
    f=open('./LabelTxt/' + filename.split('.')[0]+'.txt', "w")
    for box in root.findall(".//box"):
        bb = '0'
        for child in box:
            bb = bb + ' ' + str(child.text)
        f.writelines(bb+'\n')
    f.close()

dirs_list = 'xml107'

for file in os.listdir(dirs_list):
    saveText(file)

'''
for book in root.findall(".//box"):
    bb = []
    bb.append(0)
    for child in book:
        bb.append(child.text)
    print(bb)
    # author = book.find(".//box")
    # print(book.text.strip())


def saveText(filename):
    f=open(filename, "w")
    for book in root.findall(".//box"):
        bb = []
        bb.append(0)
        for child in book:
            bb.append(child.text)
        f.writelines(bb)
    f.close()

    with open(filename, 'w') as f:
        for book in root.findall(".//box"):
            bb = []
            bb.append(0)
            for child in book:
                bb.append(child.text)
            print(bb)

def saveImage(self):
    with open(self.labelfilename, 'w') as f:
        for bbox,bboxcls in zip(self.bboxList,self.bboxListCls):
            xmin,ymin,xmax,ymax = bbox
            # b = (float(xmin), float(xmax), float(ymin), float(ymax))
            b = (float(xmin), float(ymin), float(xmax), float(ymax))
            # bb = self.convert((self.curimg_w,self.curimg_h), b)
            f.write(str(bboxcls) + " " + " ".join([str(a) for a in b]) + '\n')
'''