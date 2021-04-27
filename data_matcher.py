import csv
# import urllib2
from PIL import Image
import requests
import re




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



                # # Save Image
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

def parse_names(name_height_map, read_path):
    file_out = open("data/final_data.txt","w")

    file1 = open(read_path, 'r')
    Lines = file1.readlines()

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



def main():

    # with open('data/all_celeb_height.csv') as f:
    #     reader = csv.DictReader(f)
    #     print(reader.fieldnames)
    name_height_map = process_csv("data/all_celeb_height.csv", "data/celeb_height_com_name.txt", "data/celeb_height_com_height.txt")
    parse_names(name_height_map, "data/list_identity_celeba.txt")


if __name__ == "__main__":
    main()