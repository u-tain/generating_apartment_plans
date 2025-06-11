import os


def get_order(path):
    '''
    Метод возвращает порядок склейки изображений
    на вход принимает путь до папки dump_match_pairs конкретной комнаты
    '''
    pairs = [item.replace('_matches.npz','') for item in os.listdir(path) if '.npz' in item]

    unique_elements = []
    last = None
    first = None

    for pair in pairs:
        el1,el2 = pair.split("_")
        if last is None:
            if '_'.join(pairs).count(el2)==1:
                last = el2
        if first is None:
            if  '_'.join(pairs).count(el1)==1:
                first = el1

        if len(unique_elements)<2:
            unique_elements.append(el1)
            unique_elements.append(el2)
        
        else:
            if first not in unique_elements and el1 not in unique_elements:
                unique_elements = [el1] + unique_elements

            if last == el2 and last not in unique_elements:
                unique_elements = unique_elements + [el2]

            elif last != el2 and el2 not in unique_elements:
                unique_elements =  unique_elements[: unique_elements.index(el1)+1] +[el2] + unique_elements[unique_elements.index(el1)+1:]

    return unique_elements
