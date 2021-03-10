# -*- coding: utf-8 -*-


# Package requirements: (Please install these packages into your computer)

# !pip install kraken
# !pip install pytesseract
# !sudo apt install tesseract-ocr
# !pip install spacy
# !python3 -m spacy download en_core_web_sm

# Imports:

from PIL import Image
import numpy as np
import pytesseract
import cv2
import spacy
import string
import kraken
from kraken import pageseg
import random
import time
import os
import sys
import xml.etree.ElementTree as ET

nlp = spacy.load("en_core_web_sm")
#filename = "/home/hp/Repositories/Subex/Subex Hackathon Dataset/images/00001.PNG"
filename = "/home/hp/Repositories/Subex/subeximages/00360.PNG"
# help(kraken)


def text(word):
    """
    Performs preprocessing on a word.
    """
    word = word.lower()
    exclude = [i for i in "!#&'()*+,-/:;<=>?@[\]^_`{|}~"]
    word = "".join(ch for ch in word if ch not in exclude)
    word = word.split()
    word = "".join(word)
    return word


def loc(x):
    """
    Getting the entity of a word using spaCy.
    """
    doc = nlp(x)
    for ent in doc.ents:
        return ent.label_
    currency_list = ["€", "£", "₹"]
    # since spaCy does not detect currecy apart from USD
    for i in currency_list:
        if i in x:
            return "MONEY"
    if x[0:2] == "rs" and len(x) > 3:
        return "MONEY"


def return_bounding_boxes(filename):
    """
    Gets the bounding boxes of any text inside an image using Kraken.
    """
    # filename = '/home/hp/Repositories/Subex/Subex Hackathon Dataset/images/00001.PNG'
    img = Image.open(filename)
    img_arr = np.asarray(img)
    try:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        pass
    ret, img_arr = cv2.threshold(img_arr, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img = Image.fromarray(img_arr)
    bounding_boxes = pageseg.segment(img.convert("1"))["boxes"]
    bounding_boxes.sort(key=lambda x: x[0])
    bounding_boxes.sort(key=lambda x: x[1])
    return bounding_boxes


def obtaining_items(sent_list):
    """
    Takes in the list obtained from get_sentences and classifies them into items.
    """
    # Lists of keywords to accept/avoid
    accepted = ["CARDINAL", "MONEY", "PERCENT"]
    avoid = [
        "bill",
        "invoice",
        "ship",
        "suite",
        "payment",
        "pay",
        "supply",
        "order",
        "bank",
    ]

    step_1 = []
    for i in sent_list:
        j = i.split()
        if len(j) > 0 and loc(text(j[-1])) in accepted:
            step_1.append(i)
        elif len(j) > 1 and loc(text(j[-2])) in accepted:
            step_1.append(" ".join(j[:-1]))
    step_2 = []
    for k in step_1:
        y = k
        k = k.split()
        flag = False
        for i in k:
            if i.lower() in avoid:
                flag = True
        if flag is False:
            step_2.append(y)
    # Using spaCy to classify sentences based on items, prices etc
    res = {}
    prev_i = 1
    try:
        for i in range(len(step_2)):
            j = step_2[i].split()
            item_dict = {}
            item_dict["Item Name"] = ""
            for k in range(len(j)):
                if text(j[k]) in ["total", "subtotal"]:
                    res["Total"] = text(j[k + 1])
                elif text(j[k]) in ["tax"]:
                    res["Tax"] = text(j[k + 1])
                elif loc(text(j[k])) is None:
                    item_dict["Item Name"] += j[k] + " "
                elif loc(text(j[k])) == "MONEY":
                    item_dict["Price"] = text(j[k])
                elif loc(text(j[k])) == "PERCENT":
                    item_dict["Tax/Discount"] = text(j[k])
                elif loc(text(j[k])) == "CARDINAL":
                    try:
                        if float(j[k]):
                            item_dict["Price"] = text(j[k])
                    except ValueError:
                        item_dict["Quantity"] = text(j[k])
            item = item_dict["Item Name"]
            flag = True
            if len(item) > 0:
                if item[0] == " ":
                    item = item[1:]
                if item[-1] == " ":
                    item = item[:-1]
            if item != "" and "total" not in item.lower():
                item_dict["Item Name"] = item
                res["Row_" + str(prev_i)] = item_dict
                prev_i += 1
        try:

            if "Price" in item_dict.keys() and res["Total"] < item_dict["Price"]:
                res["Total"] = item_dict["Price"]
            if "Total" not in res.keys():
                res["Total"] = item_dict["Price"]
        except KeyError:
            #print("Price/Total not present in Image!")
            pass

        return res
    except Exception as e:
        #print(e)
        pass


def get_sentences(filename):
    """Uses PyTesseract to get the textual data from an image and
    returns a dictionary of sentences and coordinates.
    """
    img = cv2.imread(filename)
    sent_list = []
    prev_b = 0
    box_dict = {}
    current_box = [0, 0, 0, 0]
    s = ""
    bounding_boxes = return_bounding_boxes(filename)
    for box in bounding_boxes:
        x, y, w, h = box[0], box[1], box[2], box[3]
        # b = cv2.rectangle(
        #     img.copy(), (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness=5
        # )
        x = max(0, x - 7)
        w = min(img.shape[1], w + 7)
        y = max(0, y - 7)
        h = min(img.shape[0], h + 7)
        # Cropping the image only to that bounding box
        cimg = img[y:h, x:w]
        cimg = cv2.resize(cimg, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
        # Thresholding to enhance the image features
        (_, cimg) = cv2.threshold(cimg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cimg = np.pad(cimg, ((7, 7), (7, 7)), constant_values=255)
        sent = pytesseract.image_to_string(cimg)
        # To maintain the continuity of the sentence, reading from left-to-right
        if box[0] > prev_b:
            s += sent
            prev_b = box[0]
            current_box[2] = box[2]
            current_box[3] = box[3]
        else:
            s = s.replace("\n", " ")
            s = s.replace("\x0c", "")
            sent_list.append(s)
            prev_b = box[0]
            if s != "" and s != " ":
                box_dict[s] = current_box
            s = sent
            current_box = box
    return box_dict, sent_list


def get_coordinates_of_items(img, box_dict, result_dict):
    """Iterates through the dictionary obtained from obtaining_items() and
    returns the coordinates of these items.
    """

    try:
        start = result_dict["Row_1"]["Item Name"].lower().split()
        start = random.choice(start)
        end = result_dict["Total"].lower()
    except KeyError:
        # returning the dimensions of the image itself as a contour
        return 0, 0, img.shape[0], img.shape[1]
    xmin, ymin, xmax, ymax = 99999, 99999, 0, 0
    for i in box_dict.keys():
        if start in i.lower():
            xmin, ymin = min(xmin, box_dict[i][0]), min(ymin, box_dict[i][1])
        if end in i.lower():
            xmax, ymax = max(box_dict[i][2], xmax), max(ymax, box_dict[i][3])
    return xmin, ymin, xmax, ymax


def remove_vars():
    for element in dir():

        if element[0:2] != "__":

            del globals()[element]


def pipeline():
    """Runs all the functions previously defined and shows the output."""
    img = cv2.imread(filename)
    box_dict, sent_list = get_sentences(filename)
    result_dict = obtaining_items(sent_list)
    xmin, ymin, xmax, ymax = get_coordinates_of_items(img, box_dict, result_dict)
    xmin = max(0, xmin - 10)
    xmax = min(img.shape[1], xmax + 10)
    ymin = max(0, ymin - 10)
    ymax = min(img.shape[0], ymax + 10)

    img_rect = cv2.rectangle(
        img.copy(), (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=3
    )
    for rows in result_dict.keys():
        print(rows, result_dict[rows])
    # cv2.imwrite('Screenshot.png', img_rect)
    img_rect = Image.fromarray(img_rect)
    img_rect.show()


def save_info():
    path = 'subeximages/'
    save_path = 'labelsinfo/'
    count = 0
   
    for i in os.listdir('subeximages'):
        with open (save_path + 'lastname.txt', 'r') as f:
            last_name = f.read()
        f.close()
        if i == last_name:
            pass
        else:

            last_name =  i
            with open(save_path + 'lastname.txt', 'w') as f:
                f.write(last_name)
        
            img = cv2.imread(path + i)
            box_dict, sent_list = get_sentences(path + i)
            result_dict = obtaining_items(sent_list)
            xmin, ymin, xmax, ymax = get_coordinates_of_items(img, box_dict, result_dict)
            xmin = max(0, xmin - 10)
            xmax = min(img.shape[1], xmax + 10)
            ymin = max(0, ymin - 10)
            ymax = min(img.shape[0], ymax + 10)
        
            img_rect = cv2.rectangle(
                img.copy(), (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=3
            )
            
            tree = ET.parse('labels/sample.xml')
            root = tree.getroot()
            bndbox = root.find('object').find('bndbox')
            size = root.find('size')
            bndbox.find('ymax').text = str(ymax)
            bndbox.find('xmin').text = str(xmin)
            bndbox.find('xmax').text = str(xmax)
            bndbox.find('ymin').text = str(ymin)
            size.find('width').text = str(img.shape[1])
            size.find('height').text = str(img.shape[0])
            root.find('filename').text = i
            sys.stdout = open(save_path + i.replace('PNG', 'txt'), "w")
            for rows in result_dict.keys():
                print(rows, result_dict[rows])
            sys.stdout.close()
            with open(save_path + i.replace('PNG', 'XML'), 'w') as f:
                f.write(str(ET.tostring(root))[2:-1])
            count += 1
            
            img_rect = Image.fromarray(img_rect)
            img_rect.show()


if __name__ == "__main__":
    save_info()
    # pipeline()
