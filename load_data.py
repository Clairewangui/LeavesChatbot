def get_content(filename):
    doc = os.path.join(filename)
    with open(doc, 'r') as content_file:
        lines = csv.reader(content_file,delimiter='|')
        data = [x for x in lines if len(x) == 3]
        return data
      filename = r"C:\Users\user\Desktop\DATA SCIENCE PROJECTS\BUILDING A CHATBOT\leaves.txt"
data = get_content(filename)
data
