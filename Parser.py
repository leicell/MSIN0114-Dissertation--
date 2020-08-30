#Python code to parse the pdf files

#dependencies
import os
from tika import parser
from bs4  import BeautifulSoup 
import re
import sys
from pdf2image import convert_from_path 
from pdf2image.exceptions import (
 PDFInfoNotInstalledError,
 PDFPageCountError,
 PDFSyntaxError
)
import cv2 #add later
import pytesseract 
import shutil
import pandas as pd
import numpy as np
# import json
# from  more_itertools import unique_everseen

def pdfparser(file_path,method='auto',delete_existing=True):
    ''' Use this function to automatically identify the type
    of pdf file and call corresponding functions to parse pdf and store the content
    of each page into a text file
    Parameters：
    1. method：{'auto','text' or 'image'}, default 'auto',
    Auto: choose the parsing method automatically
    Text: choose parser based on tika
    Image: choose parser based on putesseract ocr  '''

    # use tika parser to read the file first and keep the xhtml structure

    if file_path[-4:] != '.pdf':
        raise Exception("Please make sure you are parsing a pdf file")
    parsed = parser.from_file(file_path, xmlContent=True)
    raw_text = parsed['content'] #xhtml structured text
    # all the contents are stored within "page"
    soup = BeautifulSoup(raw_text,features="lxml")
    first_page = soup.find('div', attrs={ "class" : "page"}).\
    text.replace('\n','')
    if method == 'auto':
        if len(first_page)>10: #this is a text-formatted pdf
            print('parsing text-formatted pdf file')
            # write the content into a text file in 
            # the same folder 
            textpdf_to_text(file_path,raw_text)

        else:
            # parsing from a image-formatted pdf 
            # write the content into a text file in 
            # the same folder 
            print('parsing image-formatted pdf file')
            imagepdf_to_text(file_path,delete_existing)
    elif method == 'text':
        print('parsing text-formatted pdf file')
        # write the content into a text file in 
        # the same folder 
        textpdf_to_text(file_path,raw_text)
    elif method == 'image':
        # parsing from a image-formatted pdf 
        # write the content into a text file in 
        # the same folder 
        print('parsing image-formatted pdf file')
        imagepdf_to_text(file_path,delete_existing)   
def textpdf_to_text(file_path,text):
    '''Takes file_path and xhtml text as input and
    write the content of each page into a txt file'''
    soup = BeautifulSoup(text,features="lxml")
    page_list = soup.find_all('div', attrs={ "class" : "page"})
    # store all pages in a list
    page_list = [i.text for i in page_list]
    outfile=f'{file_path[:-4]}_text.txt'
    # check if the file contains bookmark
    bookmark = soup.find_all('li')
    toc = [x.text for x in bookmark if len(bookmark)>0]
    
    with open(outfile, "w+") as f:
        # Write pages
        for i,txt in enumerate(page_list):
            if i==0:
                f.write(f'<Read from Text>')
            f.write(f'<Page Start>')
            txt = txt.replace("’","'")
            f.write(txt)
            f.write(f'<Page End>')
        #Write table of contents if any
        if len(toc)>0:
            f.write(f'<PDF contains bookmark>')
            toc_string = '\n'.join(toc)
            f.write(toc_string)
    print(f'parsing finished:{file_path[:-4]}')

def imagepdf_to_text(file_path,delete_existing):
    '''Takes file_path as input, split the file into 
    pages and read text from each page write the 
    content of each page into a txt file'''
    new_folder = os.path.join(os.getcwd(),file_path[0:-4])
    try:
        os.mkdir(new_folder)
        images = convert_from_path(file_path)
        
        for i, image in enumerate(images,start=1):
            fname = f'{new_folder}/page-'+str(i)+'.png'
            image.save(fname, "PNG") 

    except:
        print(f'Folder of {file_path} already exsits')
        if delete_existing==True:
            print('Delete the exisiting folder')
            shutil.rmtree(new_folder)
            os.mkdir(new_folder)
            images = convert_from_path(file_path)

            for i, image in enumerate(images,start=1):
                fname = f'{new_folder}/page-'+str(i)+'.png'
                image.save(fname, "PNG") 
  
    outfile = f'{file_path[:-4]}_text.txt'
    maxpage = max([int(re.findall(r'[0-9]+',i)[0]) for i in os.listdir(f'{new_folder}') if i[-3:] == 'png'])

    with open(outfile, "w+") as f:
        for i in range(1, maxpage+1): 
            # reading image using opencv
            temp_filename =f'{new_folder}/page-'+str(i)+".png"
            image = cv2.imread(temp_filename)
            #converting image into gray scale image
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # converting it to binary image by Thresholding
            # this step is require if you have colored image because if you skip this part 
            # then tesseract won't able to detect text correctly and this will give incorrect result
            threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            # Recognize the text as string in image using pytesserct 
            txt = pytesseract.image_to_string(threshold_img,lang='eng') 
            # Finally, write the processed text to the file. 
            txt = txt.replace("’","'")
            if i==1:
                f.write(f'<Read from Image>')
            f.write(f'<Page Start>')
            f.write(txt)
            f.write(f'<Page End>')
    print(f'parsing finished:{file_path[:-4]}')

def text_to_dict(file_path):
    '''Use this function to read the text file and 
    store it to dictionary'''
    page_dict = {}
    file_name = file_path.split('/')[-1]
    page_dict['file_name']=file_name[0:-4]
    with open(file_path,'r') as f:
        text = f.read()
        source_type = re.findall(r'<Read from (\w+)?>',text)[0]
        page_list = re.findall(r'<Page Start>([\s\S]+?)<Page End>',text)
        toc = re.findall(r'<PDF contains bookmark>([\s\S]+)',text)
        page_dict['bookmark'] = toc
        page_dict['raw_text'] = page_list
        page_dict['source_type'] = source_type
        return page_dict

###########################################################

def expand_pagelist(df_ori):
    ''' Expand the raw_text (list type) column to rows and
    assign the page number (absolute) to each row'''
    df_expanded = df_ori.explode('raw_text').copy()
    df_expanded.reset_index(drop=True,inplace=True)
    indexstart = df_expanded.drop_duplicates(subset=['file_name'],
        keep='first').loc[:,'file_name'].copy()
    indexstart = dict(zip(indexstart,indexstart.index))
    df_expanded['tmp'] = df_expanded['file_name'].map(indexstart)
    df_expanded['page_number'] = df_expanded.index-df_expanded.tmp+1
    df_expanded.drop(columns=['tmp','row'],inplace=True)
    return df_expanded 

def extract_potential_header(df_ori):
    '''extract and clean poteintial headers from each page
    headers are in a list'''

    df = df_ori.copy()
    df['text_split_ori'] = df.raw_text.apply(
    lambda x:structure_split_by_line(x))
    df.dropna(subset=['text_split_ori'],inplace=True)
    df['text_split'] = df.text_split_ori.apply(headerlist_preprocessor)
    return df 
def structure_split_by_line(text):
    '''function within extract_header to split each page'''
    text = re.sub('\n{3,}','\n\n',text) #replace \n\n\n and above by \n\n
    text = re.sub('\n {1,}','\n',text)
    #if \n\n is not followed by a uppercase letter(or space), replace \n\n by whitespace
    text = re.sub('\n\n(?![A-Z0-9])',' ',text)
    tmp_list = text.split('\n')
    
    tmp_list = [x.strip() for x in tmp_list if len(x.strip())>0]
    tmp_list = [x for x in tmp_list if re.search(r'^[A-Z0-9]{1,}',x)] #only keep the split that starts with upper case or number
    return tmp_list

def headerlist_preprocessor(header_list):
    '''function within extract_header to preprocess header'''
    regex = r'([^ A-Za-z]+)'
    header_list = [re.sub(r' {2,}',' ',re.sub(regex,' ',h)).strip()
                   for h in header_list if len(h)<100] #keep upper case letters as they are
    header_list = [re.sub(r' [b-z] ',' ',h) for h in header_list]
    return header_list

def header_preprocessor(header_file):
    #read and preprocess the gold standard headers"
    header_keyword = pd.read_excel(header_file)
    #strip the white space after the hearder
    header_keyword['Header'] = header_keyword.apply(lambda x: x['Header'].strip().lower(), axis=1)
    header_keyword.drop_duplicates(subset='Header',keep='first',inplace=True)
    header_keyword['Length'] = header_keyword.apply(lambda x: len(x['Header']), axis=1)
    header_keyword.sort_values(axis=0, by=['Length','Header'], ascending=[False,True], inplace=True)
    header_keyword = list(header_keyword['Header'])
    regex = r'([^ A-Za-z]+)'

    standard_header = [re.sub(r' {2,}',' ',re.sub(regex,' ',h)).lower() for h in header_keyword] 
    standard_header = [re.sub(r' [b-z] ',' ',h) for h in standard_header]  

    return standard_header

def find_toc_page(df_ori,standard_header):
    '''Use this function to identify the table of contents page'''
    '''The each row in the df_ori should represents one page in a report'''
    df = df_ori.copy()
    df_file = df_ori.copy()
    df_file.drop_duplicates(subset=['file_name'],inplace=True)
    df_file = df_file.set_index('file_name')
    df_file['toc_pagekw'] = df.groupby(['file_name']).apply(lambda x:
                                                 find_toc_pagekw(x))
    df_file['toc_tmp'] = df.groupby(['file_name']).apply(lambda x:
                                                 find_toc_stheader(x,
                                                   standard_header = standard_header))
    df_file[['toc_standardheader', 'header_max_count']] = pd.DataFrame(df_file['toc_tmp'].tolist(),
                                                                       index=df_file.index)
    df_file.drop(columns='toc_tmp',inplace=True)
    df_file['toc_pagestructure'] = df.groupby(['file_name']).apply(lambda x:
                                                 find_toc_pagestructure(x))
    
    df_file['toc_page'] = df_file.loc[:,['toc_pagekw',
                                       'toc_standardheader',
                                       'toc_pagestructure']].\
                                        mode(axis=1,
                                             numeric_only=False,
                                             dropna=False)[0]
    mask = np.isnan(df_file['toc_page'])
    #if toc_page is nan but the number of standard header is greater than 20, then choose it as the toc page
    #must use loc
    df_file.loc[mask,'toc_page'] = df_file[mask].apply(lambda x:
                                                    x.toc_standardheader
                                                    if x.header_max_count>20 else np.nan, # if higher than the median count  
                                                    axis=1)
    return df_file

#use three methods to locate toc and use the majority vote to determine the final result 
#Method1 use toc keywords

def find_toc_pagekw(df_ori):
    '''Take dataframe which only contains one report and
    returns the toc page number '''
    df = df_ori.copy()
    head_search_range=200
    tail_search_range=200
    toc_page_kw = ['table of contents','contents','index','what’s inside this report']
    regex = r'|'.join(toc_page_kw)
    for i in range(0,5): 
        #note: only check first 5 pages for toc_page 
        #return only one page
        text = df.iloc[i,2] # original text
        text = re.sub(r'\n','',text.lower()) # don't need to do extra cleaning
        if re.search(regex,text[0:head_search_range]):
            return float(i+1) 
        elif re.search(regex,text[-tail_search_range:]):
            return float(i+1)
    return np.nan
#Method2:Take the page that contains the most standard header

def find_toc_stheader(df_ori,standard_header): #standard_header is a list
    df = df_ori.copy()
    result_count =[]
    for i in range(0,5):# make sure the page is in the right order
        #note: only check first 5 pages for toc_page 
        #return only one page
        text_ori = df.iloc[i,2] # 
        regex = r'([^ A-Za-z]+)'
        text_ori = re.sub(r' {2,}',' ',re.sub(regex,' ',text_ori))
        text_ori = re.sub(r' [b-z] ',' ',text_ori)
        text = text_ori.lower()
        count = 0
        for h in standard_header:
            if re.search(rf'{h}',text):
                # search if each standard_header (in lowercase) could be
                # found in text(in lowercase)
                change_once=0
                for i in range(len(re.findall(rf'{h}',text))):
                    m = re.search(rf'{h}',text)
                    text_ini = text_ori[m.start()]
                    #if the matched string starts with a Uppercase letter
                    #Then count one
                    if re.search(r'[A-Z]',text_ini) and change_once==0 :
                        count +=1
                        change_once=1
                    else:#remove the previously matched lowercase string 
                        text=text[0:m.start()]+text[m.end():]
                        text_ori=text_ori[0:m.start()]+text_ori[m.end():]
        result_count.append(count)
    # return the page where most standard headers occur    
    return float(result_count.index(max(result_count))+1),max(result_count)
#Method3:Take the page that contains table of content structure
def find_toc_pagestructure(df_ori):
    df = df_ori.copy()
    for i in range(0,5):
        #note: only check first 5 pages for toc_page 
        #return only one page
        text = df.iloc[i,2] #raw page content
         #remove long random string caused by reading the text from image
        text = re.sub(r'\S{20,}',' ',text)
        text = re.sub(r' {2,}',' ',text)
        maxpage = df_ori.shape[0]
        #the format of table of content is either page number + Title or title + page number , and followed by \n or space+\n
        num_first = re.findall(r"(?<=\n)(\d+|\d+—\d+)([ a-zA-z',\-]{3,})(?=(\n|\s\n))",text) 
        num_after = re.findall(r"(?<=\n)([ a-zA-z']{3,})(\d+|\d+—\d+)(?=(\n|\s\n))",text)
        #only take the former number if page number in the format of xx-xx
        num_first = [(int(re.findall('\d+',x[0])[0]),x[1].strip()) for x in num_first]
        num_after = [(int(re.findall('\d+',x[1])[0]),x[0].strip()) for x in num_after]
        temp = num_first + num_after 
        #filter out the none content result
        candidate = [] 
        from operator import itemgetter
        temp.sort(key=itemgetter(0))
        for j in temp: #i[0] is page number and i[1] is title
            if j[0]> maxpage:continue
            else: 
                candidate.append(j)
        if len(candidate)>5:
            return float(i+1)
    return np.nan #return nan if one page has the descirbed format

##############################################
#  Use clean_toc_page() and find_toc_headers() parse the toc page and store the potential headers with their nominal 
def clean_toc_page(text):
	# Toc page pre-cleaning
	#text -> raw_text
    if not isinstance(text,str):
        return np.nan
    else:
        text1 = re.sub(r'[^A-Za-z0-9\n]',' ',text)
        text1 = re.sub(r' [B-Zb-z] ',' ',text1)
    #1. replace all punctuation by space and replace all space greater than 2 by one;
    #replace single character as well
        text2 = re.sub(r'[0-9]{4,}',' ',text1)
    #2. replace all long numeric string by space
        text3 = re.sub('\n{2,}','\n',text2)
        text3 = re.sub('[  ]{2,}',' ',text3)
    #3. remove extra whitespace and \n
        text4 = re.sub(r'\n(?=[a-z  ]+)','',text3)
    #4. if \n is followed by a space or lowercase letter, remove \n
        text5 = re.sub(r'[a-z]{20,}','',text4) #clean long strings
        header_list = re.split('\n',text5)#split string by \n 
        stopwords = ['plc','pic','registered number','annual report',
                    'www','ltd','contents','limited']
        clean_header_list = []
        for h in header_list:
            h_lower = h.lower()
            for i,s in enumerate(stopwords):
                if s in h_lower:
                    break
                else:
                    if s not in h_lower and i == len(stopwords)-1:
                        clean_header_list.append(h.strip())

   #5. clean the if the splitted section contains any of the stop word

    return clean_header_list # clean_header_list -> a list of cleaned lines in toc_page

def find_toc_headers(df_expanded_ori,df_toc_ori,standard_header,text_pdf_mode='strict'):
    '''Parse the cleaned toc page (list) and return potential headers with page number, df_expanded_ori
    is only needed to retrieve the original text on the toc page'''
    from operator import itemgetter
    # pre-difined list/dictionary to further clean the headers while processing
    remove_list = ['Beeeeeseeeeeeees','DNAARWHN','DNAARWHN','DNAARWHN',
              'DNAARWHN','DNAARWHN','DWONAABWN','OMWNOARWNH',
              'OSNOAAR','OWONONhWN','caen','ccc','ccesee','ccsesseres',
               'ceca','cece','ceeeeee','cesses',' cies',' csc ','cscsee',
               ' ects','eecee',' eed','eee',' een','ees','ener','eseeeeeeeee',
              'occ','ooo','pwjoo','reeseeeetee','rere','seeeeeetanineeee',
               'tecscae','teeeeeeeee',' oe ',' ce ',' tea ',' cs ',' cee ',' se ',
                ' ec ',' te ',' i ',' ese ',' ee ']
    replace_dict = {'Annuat':'Annual','CONDARYN':'Contents','FIOWS':'Flows',
               'Oeferred':'Deferred','PFOVISIONS':'Provisions',
               'PraviSiONS':'Provisions','REPOMt':'Report',
                'Reportindependent':'Report independent',
                'Statementsa':'Statement','Viabilities':'liabilities',
               'fiabilities':'liabilities','incom':'income',
               'jiablities':'liabilities','liabilitie':'liabilities',
               'methodolog':'methodology','ple':'plc','rOSOVeS':'reserves',
               'remuneratio':'remuneration','tiabilities':'liabilities',
                   'tisk':'risk','REPO':'Report','fi nancial':'financial'}
    not_header_list = ['page','see','read','february','facebook']  
    # make new copies of the orginal dataset
    df_expanded = df_expanded_ori.copy()
    df_toc = df_toc_ori.copy()
    df_toc['toc_page_text'] = ''
    df_toc['toc_header_candidate'] = ''
    df_toc['toc_header_candidate'] = df_toc['toc_header_candidate'].astype('object')
    # iterate at the file level to parse the toc page
    for index,row in df_toc.iterrows():
        if np.isnan(row['toc_page']):
            df_toc.at[index,'toc_page_text'] = np.nan
            df_toc.at[index,'toc_header_candidate'] = np.nan
        else:
            headers=[]
            toc_page_number = row['toc_page']
            #extract the original text on table of content page
            text = df_expanded[(df_expanded['file_name']==index) &
                               (df_expanded['page_number']==toc_page_number)].iloc[0,2]

            clean_text_list = clean_toc_page(text) #list of clean text
            df_toc.at[index,'toc_page_text'] = clean_text_list.copy() # not all in lower case
            #Parse the toc page for header matching
            #for text pdf, use structure rules to parse first and then standard
            if df_toc.at[index,'source_type']=="Text":
            #1  try if we can extract headers with number
                nf = []
                na = []
                #print(clean_text_list)
                for i,h in enumerate(clean_text_list):
                    if text_pdf_mode == 'strict':
                        # if strict, the page number must occur at the beginning or the end of string
                        num_first = re.findall(r"(^\d+)([ a-zA-z]{3,})",h.strip()) 
                        num_after = re.findall(r"([ a-zA-z]{3,})(\d+$)",h.strip())
                    elif text_pdf_mode == 'loose':
                        num_first = re.findall(r"(\d+)([ a-zA-z]{3,})",h.strip()) 
                        num_after = re.findall(r"([ a-zA-z]{3,})(\d+)",h.strip())
                    if len(num_first)>0:
                        for j in num_first:
                            nf.append(j)
                            clean_text_list[i] = re.sub(j[0]+j[1],'',h)#Remove the matched number and string
                    if len(num_after)>0:
                        for j in num_after:
                            na.append(j)
                            clean_text_list[i] = re.sub(j[0]+j[1],'',h)#Remove the matched number and string
                nf = [(int(re.findall('\d+',x[0])[0]),x[1].strip()) for x in nf]
                na = [(int(re.findall('\d+',x[1])[0]),x[0].strip()) for x in na]
                temp = nf + na
                candidate = [] 
                temp.sort(key=itemgetter(0))
                for j in temp: #i[0] is page number and i[1] is header       
                    if j[0]> df_toc.at[index,'total_page'] or j[0]==0:continue
                    elif any([nh in j[1].lower() for nh in not_header_list ]):continue
                    elif len(j[1])>76*1.1:continue #110% of the longest header in the standard header set
                    elif ord('A')<=ord(j[1][0]) and ord('Z')>=ord(j[1][0]):
                        final_header=re.sub(r' {2,}',' ',j[1].lower())
                        candidate.append((j[0],final_header))

            # 2 try extract headers that matches standard headers 
                #print(clean_text_list) # check if the matched string has been removed
                for h in clean_text_list:
                    h = re.sub('[^A-Za-z ]','',h)
                    h = re.sub(' {2,}','',h)
                    h = h.strip()
                    if len(h)>0:
                        Up = ord('A')<=ord(h[0]) and ord('Z')>=ord(h[0])
                        if h.lower() in standard_header and Up :
                            #label all unnumberred header as 1000
                            candidate.append((1000,h.lower()))
                    else:
                        continue
                #for text pdf, use structure rules to parse first and then standard
            elif df_toc.at[index,'source_type']=="Image":
                candidate = [] 
                #clean the incorrect string caused by ocr    
                for i,h in enumerate(clean_text_list):
                    h = re.sub('|'.join(remove_list),' ',h) # only image pdf needs a second pre-clean
                    h = re.sub(' {2,}','',h)
                    h = h.strip()
                    for key in replace_dict.keys():
                        h = h.replace(key,replace_dict[key])
                    #parse header candidates that matched to standard_header
                    if len(h)>0:
                        Up = ord('A')<=ord(h[0]) and ord('Z')>=ord(h[0]) 
                        #first character of header must be upper case
                        if Up:
                            for sh in standard_header:
                                if sh in h.lower():
                                    candidate.append((1000,sh))
                                    break
                    else:continue
                
            if len(candidate)==0:
                df_toc.at[index,'toc_header_candidate'] = np.nan
                print(f'fail to find header candidate for file {index}')
            else:
                df_toc.at[index,'toc_header_candidate'] = candidate
    return df_toc
        
    
#use the following functions to match header with each page
def page_header_matcher(toc_header,text_split,raw_text,page_num,mode):
    text_split_lower = [t.lower() for t in text_split if len(t)>0]    
    for h in toc_header:
        #when assigning the header, check if the numbered page is too far way from the match page to
        #avoid miss match
        not_far = (h[0]<1000 and page_num-h[0]<6 and page_num-h[0]>0 )or h[0]==1000 
        if mode == 'strict':# match with the whole element, the initials are all uppercase
            if h[1] in text_split_lower and not_far and h!='' :
                #print(toc_header)
                toc_header.remove(h)
                return toc_header, tuple(h)
        elif mode == 'loose':
            regex = r'([^ A-Za-z]+)'
            clean_text = re.sub(regex,' ',raw_text)
            clean_text = re.sub('\n',' ',clean_text)
            clean_text = re.sub(r' {2,}',' ',clean_text)
            clean_text = re.sub(r' [B-Zb-z] ',' ',clean_text)
            #print(clean_text)
            match = re.search(h[1],clean_text.lower())
            if match!=None:
                upper = ord('A')<=ord(clean_text[match.start()]) and ord('Z')>=ord(clean_text[match.start()])
                if upper and not_far and h!='' : 
                    #print(toc_header)
                    toc_header.remove(h)
                    return toc_header, tuple(h)
    return toc_header, np.nan

def df_header_matcher(df_expanded_ori,df_toc_ori,mode='strict'):
    df_expanded = df_expanded_ori.copy()
    df_toc = df_toc_ori.copy()
    df_expanded['section_header'] = np.nan
    df_toc['indexed_page_dict'] = np.nan
    df_expanded['section_header'] = df_expanded['section_header'].astype('object')
    df_toc['indexed_page_dict'] = df_toc['indexed_page_dict'].astype('object')
    for index,row in df_toc.iterrows():
        if np.isnan(row['toc_page']):
            df_toc.at[index,'indexed_page_dict'] = np.nan
        else:
            mask = df_expanded['file_name']==index
            toc_header = row['toc_header_candidate'].copy()
#             print(index)
            for index2,row2 in df_expanded[mask].iterrows():
                if row2['page_number']>row['toc_page'] and toc_header != None:
                    toc_header, df_expanded.at[index2,'section_header'] = \
                    page_header_matcher(toc_header,row2['text_split'],row2['raw_text'],row2['page_number'],mode)
            df_tmp = df_expanded[mask].copy()
            df_tmp.dropna(subset=['section_header'],inplace=True)
            df_tmp.drop_duplicates(subset='section_header',keep='first',inplace=True)
            df_toc.at[index,'indexed_page_dict'] = dict(zip(df_tmp['page_number'],
                                                            df_tmp['section_header'])) 
        
    return df_toc    

def manual_editor(df_expanded_ori,df_toc_ori,file_name,standard_header
                  ,show_result=False,parser_auto=True,cha_page_to=None,parser='strict',matcher='strict',text_auto=False):
    ''' Use this function to mannually switch parsing or matching method'''               
    '''Input dataframe, the file needs to be modified and modify level
    There three independent option for modification: toc_page, toc_parser(only for text pdf)
    ,and toc matcher (for both text pdf and image pdf)'''
    '''Parameters: 
    1.show_result:{True or False},default False
    If True, show the edited index dict before and after mannual editing 
    2.parser_auto:{True or dict},default True
    dict: Enter a mannual toc as {actual pagenumber(nominal pagenumber,header)}
    3.page: {int or None}, default None 
    Change the toc_page to the input number, if None is passed, no change
    4.parser: {'strict' or 'loose'},default 'strict'
    Change the toc parse mode (will only affect text pdf).
    "strict": the number in the numbered header must appear at the beginning or end of the entire string
    "loose": the number in the numbered header can appear in the middle of a string
    5.matcher:{'strict','loose'},default 'strict'
    Change the toc matcher mode (will affect both text and image pdf)
    'strict': the candidate header must match the whole string
    'loose': the candidate header can match a substring of the whole string
    6. text_auto:{True or False},default False
    'True': sikp the matcher function and assign the whole indexed toc for text pdf directly
    '''
    df_expanded = df_expanded_ori.copy()
    df_toc = df_toc_ori.copy()
    df_toc['indexed_page_dict'] = df_toc['indexed_page_dict'].astype('object')
    #08/21
    if 'toc_header_candidate' not in df_toc.columns:
        df_toc['toc_header_candidate'] = ''
        df_toc['toc_header_candidate'] = df_toc['toc_header_candidate'].astype('object')
    #0821
    if isinstance(file_name,str):
        file_name = [file_name] # turn single file to a list
    for f in file_name:
        indexed_page_dict = np.nan # initialize indexed_page_dict with np.nan
        if show_result == True:
            print(f'change toc for: {f}')
            print(f"The original toc is: {df_toc.at[f,'indexed_page_dict']}")
        else: pass
        
        if parser_auto!=True:
            df_toc.at[f,'indexed_page_dict'] = parser_auto
        else:
            if cha_page_to == None:
                page = df_toc.at[f,'toc_page'] #toc_page
            else:
                df_toc.at[f,'toc_page'] = cha_page_to
            # same preprocessing for both Text and Image    
            tmp_toc1 = df_toc.loc[[f]].copy() #slice only the relevent part of df_toc as a dataframe 
            tmp_toc1 = find_toc_headers(df_expanded,tmp_toc1,standard_header,text_pdf_mode=parser)
            #08/21
            # print('xx-xx')
            # print(tmp_toc1.at[f,'toc_header_candidate'])
            #08/21
            df_toc.at[f,'toc_header_candidate'] = tmp_toc1.at[f,'toc_header_candidate']
            if df_toc.at[f,'source_type'] == 'Text':
                if text_auto == True: # match indexed headers altogether

                    header_candidate = [i for i in df_toc.at[f,'toc_header_candidate'] if i[0]<1000]
                    for i in header_candidate: 
                        index_start_page = 1 
                        page_shift = None
                        while index_start_page <= i[0]+page: 
                            firstheader_true_pagenum = index_start_page+i[0]-1
                            if firstheader_true_pagenum <= page: 
                                #the actual page number where the first header appears must 
                                #be larger than toc_page
                                index_start_page += 1
                                continue
                            page_mask = (df_expanded['file_name'] == f) &\
                            (df_expanded['page_number']==firstheader_true_pagenum) 
                            #start searching for the page where first indexed header appears and assuming index start
                            #from page one
                            text_split = df_expanded.loc[page_mask,'text_split'].iloc[0]
                            text_split_lower = [t.lower() for t in text_split if len(t)>0]
                            for t in text_split_lower: # match with the any substring of the element
                                #print(t)
                                if  i[1] in t :
                                    page_shift =  index_start_page-1
                                    break
                            if page_shift != None:
                                break
                            index_start_page += 1
                        try:
                            indexed_page_dict = {i[0]+page_shift:i for i in header_candidate}
                            break
                        except:
                            pass
                    df_toc.at[f,'indexed_page_dict'] = indexed_page_dict
                elif text_auto == False:
                    tmp_toc2 = df_toc.loc[[f]].copy() #slice only the relevent part of toc
                    tmp_toc2 = df_header_matcher(df_expanded,tmp_toc2,mode=matcher)
                    df_toc.at[f,'indexed_page_dict'] = tmp_toc2.at[f,'indexed_page_dict']
            elif df_toc.at[f,'source_type'] == 'Image':
                tmp_toc2 = df_toc.loc[[f]].copy() #slice only the relevent part of toc
                tmp_toc2 = df_header_matcher(df_expanded,tmp_toc2,mode=matcher)
                df_toc.at[f,'indexed_page_dict'] = tmp_toc2.at[f,'indexed_page_dict']
        if show_result == True:
            print(f"The new toc is: {df_toc.at[f,'indexed_page_dict']}")
        else:pass
    return df_toc



# auto-matcher pipeline: 
def auto_parser_matcher(df_expanded_ori,df_toc_ori,standard_header):
    '''Use all different methods to parse/match toc, generate features and 
    select the best result based on pre-determined rule '''
    df_toc = df_toc_ori.copy()
    df_expanded = df_expanded_ori.copy()
    df_toc0 = find_toc_headers(df_expanded,df_toc,standard_header,text_pdf_mode='strict')
    
    #1 strict parser and strict matcher
    df_toc1 = df_header_matcher(df_expanded,df_toc0,mode='strict')
    df_toc1.rename(columns={'indexed_page_dict':'str_par_str_mat'},inplace=True)
    
    #2 strict parser and loose matcher
    df_toc2 = df_header_matcher(df_expanded,df_toc0,mode='loose')
    df_toc2.rename(columns={'indexed_page_dict':'str_par_loo_mat'},inplace=True)
    
    #3  loose parser and strict matcher
    df_toc00 = find_toc_headers(df_expanded,df_toc,standard_header,text_pdf_mode='loose')
    df_toc3 = df_header_matcher(df_expanded,df_toc00,mode='strict')
    df_toc3.rename(columns={'indexed_page_dict':'loo_par_str_mat'},inplace=True)
    
    #4 loose parser and strict matcher
    df_toc4 = df_header_matcher(df_expanded,df_toc00,mode='loose')
    df_toc4.rename(columns={'indexed_page_dict':'loo_par_loo_mat'},inplace=True)
    
    #For text only
    text_file_list = df_toc[df_toc['source_type']=='Text'].index.to_list()
    
    #5. strict parser auto_text=True
    df_toc1.rename(columns={'str_par_str_mat':'indexed_page_dict'},inplace=True)
    df_toc5 = manual_editor(df_expanded,df_toc1,text_file_list,standard_header,
                            parser='strict',text_auto=True)
    df_toc5.rename(columns={'indexed_page_dict':'str_par_auto'},inplace=True)
    
    #6. loose parser auto_text=True
    df_toc3.rename(columns={'loo_par_str_mat':'indexed_page_dict'},inplace=True)
    df_toc6 = manual_editor(df_expanded,df_toc1,text_file_list,standard_header,
                            parser='loose',text_auto=True)
    df_toc6.rename(columns={'indexed_page_dict':'loo_par_auto'},inplace=True)
    df_toc1.rename(columns={'indexed_page_dict':'str_par_str_mat'},inplace=True)
    df_toc3.rename(columns={'indexed_page_dict':'loo_par_str_mat'},inplace=True)
    df_toc_all = df_toc1.copy()
    df_toc_all['str_par_loo_mat'] = df_toc2['str_par_loo_mat']
    df_toc_all['loo_par_str_mat'] = df_toc3['loo_par_str_mat']
    df_toc_all['loo_par_loo_mat'] = df_toc4['loo_par_loo_mat']
    df_toc_all['str_par_auto'] = df_toc5['str_par_auto']
    df_toc_all['loo_par_auto'] = df_toc6['loo_par_auto']
    toc_tmp = pd.melt(df_toc_all.reset_index(),id_vars=['file_name','source_type','toc_header_candidate','toc_page'], 
                  value_vars=['str_par_str_mat','str_par_loo_mat','loo_par_str_mat','loo_par_loo_mat','str_par_auto','loo_par_auto'],
        var_name='parser_method', value_name='indexed_page_dict')
    #08/21
    toc_tmp = toc_tmp[toc_tmp['indexed_page_dict']!={}]
    #08/21
    toc_tmp['all_header_count'] = toc_tmp.apply(lambda x: len(x['indexed_page_dict']) if isinstance(x['indexed_page_dict'],dict)
                                                     else np.nan, axis=1)
    toc_tmp['indexed_header_count'] = toc_tmp.apply(lambda x: count_indexed_header(x['indexed_page_dict']) if isinstance(x['indexed_page_dict'],dict)
                                                         else np.nan, axis=1)
    toc_tmp['wrong_order_level'] = toc_tmp.apply(lambda x: wrong_order_level(x['source_type'],
    x['indexed_page_dict'],x['toc_header_candidate']) if isinstance(x['indexed_page_dict'],dict) else np.nan, axis=1)
    toc_tmp['avg_wrong'] = toc_tmp['wrong_order_level']/toc_tmp['all_header_count']
    toc_tmp['indexed_all'] = toc_tmp['indexed_header_count']/toc_tmp['all_header_count']
    toc_tmp['page_diff_var'] = toc_tmp.apply(lambda x: page_diff_var(x['source_type'],x['indexed_page_dict'])
                                                          if isinstance(x['indexed_page_dict'],dict)
                                                         else np.nan, axis=1)
    toc_tmp.fillna(0,inplace=True)
    x = pd.DataFrame()
    x['result'] = toc_tmp.groupby('file_name').apply(lambda x: find_best_method(x) )
    x = pd.DataFrame(x.result.to_list(),index=x.index,columns=['file_name','method','indexed_page_dict'])
    x.set_index('file_name',inplace=True)
    df_toc1.drop('str_par_str_mat',axis=1,inplace=True)
    df_toc_final = df_toc1.merge(x, right_index = True, left_index = True)
    df_toc_final = df_toc_final.loc[:,['source_type','total_page',
                                       'firm_name','year',
                                       'toc_page','toc_header_candidate','method', #08/21
                                       'indexed_page_dict']]
    return df_toc_final


#feature generation functions to indentify best parsing result

#Feature 1
def count_indexed_header(indexed_page_dict):
    count = 0
    for k,v in indexed_page_dict.items():
        if v[0]<1000:
            count += 1
    return count

#Feature 2
def wrong_order_level(source_type,indexed_page_dict,toc_header_candidate): 
    #only for the evaluation of image pdf 
    ran_level=0
    if source_type == 'Image': 
        ori_order_dict = {toc_header_candidate[i][1]:i for i in range(len(toc_header_candidate))}
        new_order_dict = {}
        pos = 0
        for k,v in indexed_page_dict.items():
            new_order_dict[v[1]] = pos
            ran_level += abs(pos-ori_order_dict[v[1]])
            pos += 1
    return ran_level

#Feature 3
def page_diff_var(source_type,indexed_page_dict):
    page_diff_var = 0
    if source_type == 'Text':
        page_actual_diff = [k-v[0] for k,v in indexed_page_dict.items() if v[0]<1000]
        if len(page_actual_diff)>2:
            mean = sum(page_actual_diff)/len(page_actual_diff)
            tmp = [(diff-mean)**2 for diff in page_actual_diff]
            page_diff_var = sum(tmp)/len(tmp) 
            return page_diff_var
        else:
            return page_diff_var
    return page_diff_var         

def find_best_method(df):
    if df.iloc[0,1] == 'Text': 
        #select the 4 with the largest header counts
        df = df.nlargest(4,'all_header_count', keep='first')
        #method: first find the highest indexed/all ratio(can't be too small) and then find the one with lowest page_diff_var ;
        all_header_count = df['all_header_count'].max()
        page_diff_var = df['page_diff_var'].min()
        highest_indexed_all = df['indexed_all'].max()
        #tmp = df[df['indexed_all']==highest_indexed_all].copy()
        tmp = df[df['indexed_all']==highest_indexed_all].copy()
        #if multiple max, return the one(s) with the lowest page_diff_var,and the highest header
        page_diff_var = tmp['page_diff_var'].min()
        tmp = df[df['page_diff_var']==page_diff_var].copy()
        index = tmp['all_header_count'].idxmax(axis=0, skipna=True)
        if all_header_count<df.loc[index,'all_header_count']*1.5:
            return (df.loc[index,'file_name'],df.loc[index,'parser_method'],
               df.loc[index,'indexed_page_dict'])
        else: #if header count is too small
            df2 = df[df['indexed_all']<highest_indexed_all].copy()
            highest_indexed_all = df2['indexed_all'].max()
            if pd.isna(highest_indexed_all): 
                # if all indexed/all ratios are the same,return the one with highest headcount number
                index = df['all_header_count'].idxmin(axis=0, skipna=True)
                return   (df.loc[index,'file_name'],df.loc[index,'parser_method'],
                   df.loc[index,'indexed_page_dict'])          
            tmp = df[df['indexed_all']==highest_indexed_all].copy()
            #index = tmp['all_header_count'].idxmax(axis=0, skipna=True)
            index = tmp['page_diff_var'].idxmin(axis=0, skipna=True)
            if all_header_count<df.loc[index,'all_header_count']*1.5:
                # if all_header_count is too small then drop it and find
                # the second best
                return (df.loc[index,'file_name'],df.loc[index,'parser_method'],
                   df.loc[index,'indexed_page_dict'])
            else:
                index = index = df['all_header_count'].idxmin(axis=0, skipna=True) #use the one with highest header count
                return (df.loc[index,'file_name'],df.loc[index,'parser_method'],
                   df.loc[index,'indexed_page_dict'])
    elif df.iloc[0,1] == 'Image': 
        lowest_avg_wrong = df['avg_wrong'].min()
        tmp = df[df['avg_wrong']==lowest_avg_wrong].copy()
        index = tmp.index[0] # by default use the first occurance
        #print(df.loc[index,'file_name'])
        return (df.loc[index,'file_name'],df.loc[index,'parser_method'],
               df.loc[index,'indexed_page_dict'])

def retrieve_header(file_name,page_number,total_page,df_toc_final):
    try:
        page_dict = df_toc_final.loc[file_name,'indexed_page_dict'].copy()
    except:
        print(file_name)
    headerpage_list = [k for k in page_dict.keys()]
    headerpage_list.insert(0,0)
    headerpage_list.append(total_page+1)
    page_dict[0] = (0,'before main content')
    for i in range(len(headerpage_list)-1):
        if page_number>= headerpage_list[i] and page_number<headerpage_list[i+1]:
            p = headerpage_list[i]
            return page_dict[p][1]
        else:continue