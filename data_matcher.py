import csv
# import urllib2
from PIL import Image
import requests
import re

import os 
import matplotlib.pyplot as plt

def process_gay_csv(gay_csv_path):
        read_file_path = gay_csv_path
        # Name to height map init
        name_gay_map = {}
        # Image index
        ind = 0

        with open(read_file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Get celeb name
                name = row['\ufeffName'].strip()
                name_key = row['\ufeffName'].replace(" ", "").lower()

                # Skip if redundant occurance 
                if name_key in name_gay_map.keys():
                    continue

                # Add to map
                name_gay_map[name_key] = "gay"
                ind += 1
        return name_gay_map


def process_csv(height_csv_path, save_name_path, save_height_path):
        read_file_path = height_csv_path
        # Name to height map init
        name_height_map = {}
        # Image index
        ind = 0

        file_nm = open(save_name_path,"w")
        file_ht = open(save_height_path,"w")

        with open(read_file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Get celeb name
                name = row['Title_URL'].strip()
                name_key = row['Title_URL'].replace(" ", "").lower()

                # Skip if redundant occurance 
                if name_key in name_height_map.keys():
                    continue

                # Get height
                height_str = re.search(r'\((.*?)\)',row['Title']).group(1).split(' ')[0]
                height = float(re.search(r'\((.*?)\)',row['Title']).group(1).split(' ')[0])
                
                # Add to map
                name_height_map[name_key] = [height, name]

                # # Saving images from celebrityheights.com
                # image_id = ind
                # image_id_str = str(image_id)
                # # adhere to the image path format (the image number is 6 digits long)
                # while len(image_id_str) < 6:
                #     image_id_str = '0' + image_id_str
                # img_url = row['Image']
                # if img_url != '':
                #     im = Image.open(requests.get(img_url, stream=True).raw)
                #     # Save img to 
                #     im = im.save("data/celeb_height_web/_" + image_id_str + ".jpg")

  
                # \n is placed to indicate EOL (End of Line)
                file_nm.write(name + "\n")
                file_ht.write(height_str + "\n")
                ind += 1
        # print(name_height_map)
        return name_height_map

def parse_gay_names(name_gay_map, read_path, out_file_dir):

    file_out = open(out_file_dir,"w")

    file_celeba = open(read_path, 'r')
    Lines = file_celeba.readlines()

    cnt = 0
    gay_cnt = 0
    celeb_map = {}
    gay_map = {}
 
    for line in Lines:
        name = line.split()[1].lower().replace("_", "")
        celeb_map[name] = ""
        if name in name_gay_map.keys():
            # (file name, dict key, actual name, gay or not)
            l = line.split()[0] + " " + name + " " + line.split()[1] + " " + "gay" + "\n"
            file_out.write(l)
            cnt += 1
            gay_cnt += 1
            gay_map[line.split()[1]] = ""
        else:
            # (file name, dict key, actual name, gay or not)
            l = line.split()[0] + " " + name + " " + line.split()[1] + " " + "straight" + "\n"
            file_out.write(l)
            cnt += 1


    print("Gay images in CelebA: ", gay_cnt)
    print("CelebA Size: ", len(Lines))
    print("Number of gays in dataset: ", len(gay_map))
    print("Number of total celebs: ", len(celeb_map))
    print("Gay image Rate: ", float(gay_cnt / len(Lines)))
    print("Gay rate: ", len(gay_map)/len(celeb_map))

    print(gay_map.keys())

def parse_names(name_height_map, read_path, out_file_dir):
    file_out = open(out_file_dir,"w")

    file_names = open(read_path, 'r')
    Lines = file_names.readlines()

    cnt = 0
 
    for line in Lines:
        name = line.split()[1].lower().replace("_", "")
        if name in name_height_map.keys():
            # (file name, dict key, actual name, height)
            l = line.split()[0] + " " + name + " " + line.split()[1] + " " + str(name_height_map[name][0]) + "\n"
            file_out.write(l)
            cnt += 1

    print("Matches between CelebA and CelebrityHeight.com: ", cnt)
    print("CelebA Size: ", len(Lines))
    print("Match Rate: ", float(cnt / len(Lines)))

def process_vip_data(vip_img_dir, vip_csv_dir, img_save_dir, csv_save_dir, part_list_dir):
    
    # Starting index
    ind = 202600

    # open files in append mode
    file_final = open(csv_save_dir,'a')
    file_part = open(part_list_dir,'a')

    # img number to height map
    img_height_map = {}
    with open(vip_csv_dir, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_path = row['image']
            height = str(float(row['height']) * 100)
            # Add to map
            img_height_map[img_path] = height

    # Separate male and female files
    f_list = []
    m_list = []
    entries = os.listdir(vip_img_dir)
    for entry in entries:
        # print(entry)
        if entry[0] == "f":
            f_list.append(entry)
        else:
            m_list.append(entry)

    # Devide into train data and val data, and save to files
    m_train_len = int(len(m_list) * (4 / 5))
    for i, item in enumerate(m_list):
        img_file_name = str(ind) + ".jpg"
        # Save img to img_align
        im = Image.open(vip_img_dir + item)
        im = im.save(img_save_dir + img_file_name)
        # Save new info to final_data.txt
        l = img_file_name + " " + "dumbass" + " " + "Dumb Ass" + " " + str(img_height_map[item[0:-4]]) +"\n"
        file_final.write(l)

        # Save to partition txt
        if i < m_train_len:
            l = img_file_name + " " + "0\n"
            file_part.write(l)
        else:
            l = img_file_name + " " + "1\n"
            file_part.write(l)
        ind += 1

    f_train_len = int(len(f_list) * (4 / 5))
    for i, item in enumerate(f_list):
        img_file_name = str(ind) + ".jpg"
        # Save img to img_align
        im = Image.open(vip_img_dir + item)
        im = im.save(img_save_dir + img_file_name)

        # Save new info to final_data.txt
        l = img_file_name + " " + "dumbass" + " " + "Dumb_Ass" + " " + str(img_height_map[item[0:-4]]) + "\n"
        file_final.write(l)

        # Save to partition txt
        if i < f_train_len:
            l = img_file_name + " " + "0\n"
            file_part.write(l)
        else:
            l = img_file_name + " " + "1\n"
            file_part.write(l)
        ind += 1
        
def main():
    
    # with open('data/gaylist.csv') as f:
    #     reader = csv.DictReader(f)
    #     print(reader.fieldnames)

    # name_height_map = process_csv("data/all_celeb_height.csv", "data/celeb_height_com_name.txt", "data/celeb_height_com_height.txt")
    # parse_names(name_height_map, "data/list_identity_celeba.txt", "data/final_data.txt")
    # process_vip_data("data/vip/data/", "data/vip/annotation.csv", "data/img_align_celeba_extend/", "data/final_data.txt", "data/list_eval_partition.txt")
    name_gay_map = process_gay_csv("data/gaylist.csv")
    parse_gay_names(name_gay_map, "data/list_identity_celeba.txt", "data/final_gay_data.txt")

if __name__ == "__main__":
    main()