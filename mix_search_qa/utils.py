import csv
import re
import os
from bs4 import BeautifulSoup
csv.field_size_limit(int(4e6))

# 处理表格每一行文本
def process_row(row):
    row = [i.replace(" ", "").replace("\xa0", "").replace("\n","").replace("\u3000", "").replace("。","") for i in row]
    row = ["None" if element in ["", None] else element for element in row]
    # prefix = "**" if i<n_col else "*"
    return "|" + "|".join(row)+"|\\n" 

# 处理表格
def get_strlist_from_table(table):
    rows = table.find_all('tr')  # Find all <tr> elements within the table
    table_list = []
    n_cols = [] 
    for row in rows:
        row_list = []
        cells = row.find_all('td')  # Find all <td> elements within each row
        for cell in cells:
            row_list.append(cell.get_text())
        table_list.append(row_list)
        n_cols.append(len(row_list))
    n_col = n_cols.index(max(n_cols))
    table_list[0] = [i for i in table_list[0]]
    # table_list = [i for i in table_list if len(i)==n_cols]
    # 表格的标题前加上**，为以后正则可以检测到
    title = "**"
    result = []
    for i in range(len(table_list)):
        row = table_list[i]
        if not row:
            break
        if i<n_col or (n_col==0 and i==0):
            title += process_row(row)
        else:
            result.append("*"+process_row(row))
    if title:
        result.insert(0, title)
    return result
    # return [process_row(table_list[i], i, n_col) for i in range(len(table_list))]


def remove_html_tags_and_split(raw_text):
    soup = BeautifulSoup(raw_text, 'html.parser')
    tables = soup.find_all('table')

    paragraphs = []
    # 如果有表格文本的情况下：
    if tables:
        for i, table in enumerate(tables):
            if i == 0:
                temp = [elem.get_text() for elem in table.find_previous_siblings() 
                        if elem.name != 'table'
                        and elem.name=="p"][::-1]
                paragraphs+=temp
            try:
                paragraphs+=get_strlist_from_table(table)
            except:
                pass

            if i < len(tables) - 1:
                # temp = [elem.get_text() for elem in table.find_next_siblings() 
                #         if elem.name != 'table' and elem != tables[i+1]
                #         and elem.name=="p"]
                temp = [elem.get_text() for elem in table.find_next_siblings() 
                        if elem.name != 'table' and elem != tables[i+1]]
                paragraphs+=temp
            if i == len(tables) - 1:
                # temp = [elem.get_text() for elem in table.find_next_siblings() 
                #         if elem.name != 'table'
                #         and elem.name=="p"]
                temp = [elem.get_text() for elem in table.find_next_siblings() 
                        if elem.name != 'table']
                paragraphs+=temp
    # 如果没有表格的情况下
    else:
        # temp = [p.get_text().strip() for p in soup.find_all('p')]
        temp = [p.get_text().strip() for p in soup.find_all(lambda tag: tag.name != 'table')]
        paragraphs+=temp
    # split based on \n and \u3000\u3000
    paragraphs = [p.replace("\u3000\u3000", "\n") for p in paragraphs]
    paragraphs = [p.replace("。", "。\n") for p in paragraphs]
    paragraphs = [p.replace("；", "；\n") for p in paragraphs]
    new_paragraphs = []
    for p in paragraphs:
        if "\n" in p:
            new_paragraphs += p.split("\n")
        else:
            new_paragraphs.append(p) 
    # remove \xa0 and \u3000
    new_paragraphs = [p.replace("\xa0", "").replace("\u3000", "") for p in new_paragraphs]
    # remove ""
    new_paragraphs = [p.strip() for p in new_paragraphs]

    # new_paragraphs = [p for p in new_paragraphs if p]
    new_paragraphs = [p for p in new_paragraphs if len(p)>1 and p != "答："]

    new_paragraphs = list(dict.fromkeys(new_paragraphs))
    return new_paragraphs



PATTERN_VALUES = [r"^第[一二三四五六七八九十百零]{1,5}[章]",
                  r"^第[一二三四五六七八九十百零]{1,5}[条]",
                  r"^第[一二三四五六七八九十百零]{1,5}[款]",
                  r"^第[一二三四五六七八九十百零]{1,5}[节]",
                  r"^第[一二三四五六七八九十百零]{1,5}[点]",
                  r"^[一二三四五六七八九十百零]{1,5}[是]",
                  r"^（[一二三四五六七八九十百零]{1,5}）",
                  r"^\([一二三四五六七八九十百零]{1,5}\)",
                  r"^[一二三四五六七八九十百零]{1,5}、",
                  r"^第[一二三四五六七八九十百零]{1,5}步",
                  
                  r"^[1234567890]{1,3}\.",
                  r"^（[1234567890]{1,3}）",
                  r"^\([1234567890]{1,3}\)",
                  r"^[1234567890]{1,3}\、",
                  r"^Q[1234567890]{1,3}",
                  r"^方式[1234567890]{1,3}",
                  r"问：",
                  r"答：",

                  r"^[１２３４５６７８９０]{1,3}\.",
                  r"^（[１２３４５６７８９０]{1,3}）",
                  r"^\([１２３４５６７８９０]{1,3}\)",
                  r"^[１２３４５６７８９０]{1,3}\、",
                  r"^Q[１２３４５６７８９０]{1,3}",

                  r"^[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂]",
                  r"^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]",
                  r"^释义",
                  r"^◆",
                  r"^\*\*",
                  r"^\*",
                  r"^——(.*?)--$",
                  r"^附录",
                  r"^附件",
                  r"^附则",
                  r"：$",
                  r"？$",
                  r"[\u4e00-\u9fff1234567890a-z！）)]$"
                  ]
PATTERN_DICTS = {f"pattern_{i}":PATTERN_VALUES[i] for i in range(len(PATTERN_VALUES))}



def topic_split(paragraphs):
    if not paragraphs:
        return []
    for i in range(len(paragraphs)):
        for pattern_name, pattern in PATTERN_DICTS.items():
            result = re.search(pattern, paragraphs[i])
            if result:
                matched = result.group(0).strip()
                remained = re.sub(pattern, "", paragraphs[i])
                # paragraphs[i] = [pattern_name, matched,  remained]
                paragraphs[i] = [pattern_name, paragraphs[i]]
                break
            else:
                continue
    # return flatten_list(paragraphs)
    return paragraphs





def process_content(content_id, content_title, content):
    
    content_title = content_title#.replace(" ", "").replace("/", "")[:50] 
    # html文本，去除tag，处理后转成列表，列表里正常文本是string，如果是标题则是列表
    # splitted是一个list of (list or string)
    splitted = topic_split(remove_html_tags_and_split(content))
    if splitted is []:
        return []
    
    result_list = []
    past_pattern_ids = []
    
    for i in range(len(splitted)):

        curr_paragraph = splitted[i]
        past_patterns = [past_pattern_ids[j][0] for j in range(len(past_pattern_ids))]
        past_ids = [past_pattern_ids[j][1] for j in range(len(past_pattern_ids))]

        # str不需要处理past_pattern_ids, 不需要有children
        if isinstance(curr_paragraph, str):
            curr_dict = {"parents":[content_title], "content":curr_paragraph, "children":[]}
            if past_pattern_ids==[]:
                pass
            else:
                if i==0 and curr_paragraph==curr_dict["parents"][-1]:
                    continue
                curr_dict["parents"]+=[splitted[k][-1] for k in past_ids]

        # 如果不是任何标题，不是list type，的情况：
        if isinstance(curr_paragraph, list):
            curr_pattern = curr_paragraph[0]
            curr_content = curr_paragraph[-1]
            curr_dict = {"parents":[content_title], "content":curr_content, "children":[]}
            # 如果match到了附录，附件，附则，就清除parent
            if curr_dict["parents"] and curr_pattern in [f"pattern_{k}" for k in [26+2,27+2,28+2,29+2]]:
                curr_dict["parents"] = curr_dict["parents"][1:]
            # 如果在遇到第一个pattern之前
            if past_pattern_ids==[]:
                pass
            elif i==0 and curr_content==curr_dict["parents"][-1]:
                pass
            # 如果之前出现过这个pattern
            elif curr_pattern in past_patterns:
                index = past_patterns.index(curr_pattern)
                past_ids = past_ids[:index]
                curr_dict["parents"]+=[splitted[k][-1] for k in past_ids]
                past_pattern_ids = past_pattern_ids[:index]
                
            # 如果是新的pattern
            else:
                curr_dict["parents"]+=[splitted[k][-1] for k in past_ids]
            # 保存当前pattern，除了“释义”的情况
            if curr_pattern != "pattern_25":
                past_pattern_ids.append((curr_pattern, i))
                past_patterns = [past_pattern_ids[j][0] for j in range(len(past_pattern_ids))]
            # 寻找childrens 
            children = []
            children_type = None
            for j in range(len(splitted)):
                if j<=i:
                    continue
                # 如果是任何标题，是list type，的情况：
                if isinstance(splitted[j], list):
                    if splitted[j][0] in past_patterns:
                        break
                    if children_type is not None and children_type != splitted[j][0]:
                        continue
                    children.append(splitted[j][-1])
                    children_type = splitted[j][0]
                    # print("== list", paragraphs2[j][0], children_type)
                else:
                    if children_type is not None and children_type != "str":
                        continue
                    children.append(splitted[j])
                    children_type = "str"
                    # print("== str", paragraphs2[j], children_type)
            if children:
                curr_dict["children"] += children
        curr_dict["content_title"] = content_title
        curr_dict["id"] = content_id
        result_list.append(curr_dict)
    if result_list:
        return result_list
    else:
        return []

import hashlib
def calculate_hash(text):
    hash_obj = hashlib.sha256()
    hash_obj.update(text.encode('utf-8'))
    return hash_obj.hexdigest()
