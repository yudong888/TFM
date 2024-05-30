import xml.etree.ElementTree as ET 
import shutil, os

'''
tree = ET.parse('test.xml')
root = tree.getroot()
# f=open('test.txt')
for object in root.findall("object"):
    if object[0].text == 'Bad': head = '0'
    if object[0].text == 'Middle': head = '1'
    if object[0].text == 'Good': head = '2'
    print('\n' + head)
    for child in object[7][1]:
        print(child.text)
    # print(head, object[7][1][0].text, object[7][1][1].text, object[7][1][2].text, object[7][1][3].text)
    # f.writelines(object+'\n')
# f.close()
'''

def saveText(filename): 
    tree = ET.parse('./' + dirs_list + '/' + filename)
    root = tree.getroot()
    f=open('./Labels/' + filename.split('.')[0]+'.txt', "w")
    for object in root.findall("object"):
        if object[0].text == 'Pig': head = '0'
        if object[0].text == 'Middle': head = '1'
        if object[0].text == 'Good': head = '2'
        bb = head
        for child in object[7][1]:
            bb = bb + ' ' + str(child.text)
        f.writelines(bb+'\n')
    f.close()

dirs_list = 'xml'

for file in os.listdir(dirs_list):
    saveText(file)
