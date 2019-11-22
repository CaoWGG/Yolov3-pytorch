from collections import OrderedDict
def parse(cfgfile):
    def erase_comment(line):
        line = line.split('#')[0]
        return line
    blocks = []
    fp = open(cfgfile, 'r')
    block =  None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = OrderedDict()
            block['type'] = line.lstrip('[').rstrip(']')
            # set default value
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            line = erase_comment(line)
            key,value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = eval(value) if str.isnumeric(value) else value
        line = fp.readline()
    if block:
        blocks.append(block)
    fp.close()
    return blocks

def parse_cfg(cfgfile):
    blocks = parse(cfgfile)
    need_save = []
    print('layer     filters    size              input                output')
    for ind,block in enumerate(blocks):
        if block['type'] == 'net':

            block['momentum'] = float(block['momentum'])
            block['decay'] = float(block['decay'])
            block['width'] = int(block['width'])
            block['height'] = int(block['height'])
            block['filters'] = int(block['channels'])
            block['batch'] = int(block['batch'])
            block['stride'] = []

        elif block['type'] == 'convolutional':
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            batch_normalize = int(block['batch_normalize'])
            pad = kernel_size//2
            width = int((blocks[ind - 1 ]['width'] + 2*pad - kernel_size)/stride + 1)
            height = int((blocks[ind - 1]['height'] + 2*pad - kernel_size)/stride + 1)
            block['filters'] = filters
            block['size'] = kernel_size
            block['stride'] = stride
            block['pad'] = is_pad
            block['batch_normalize'] = batch_normalize
            block['width'] = width
            block['height'] = height
            block['in_channel'] = blocks[ind - 1]['filters']
            print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d   ' % (
            ind, 'conv', filters, kernel_size, kernel_size, stride, blocks[ind - 1]['width'], blocks[ind - 1]['height'], block['in_channel'], width, height, filters))

        elif block['type'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            width = blocks[ind - 1]['width']/stride
            height = blocks[ind - 1]['height']/stride
            block['size'] = pool_size
            block['stride'] = stride
            block['width'] = width
            block['height'] = height
            block['in_channel'] = blocks[ind - 1]['filters']
            block['filters'] = blocks[ind - 1]['filters']
            print('%5d %-6s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'max', pool_size, pool_size, stride, blocks[ind - 1]['width'], blocks[ind - 1]['height'], blocks[ind - 1]['filters'], width, height, block['filters'] ))

        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            layers = [int(i)+1 if int(i) > 0 else int(i)+ind for i in layers]
            print('%5d %-6s ' % (ind, 'route'),end='')
            print(', '.join(list(map(str,layers))))
            block['in_channel'] = [blocks[f_id]['filters'] for f_id in layers]
            block['filters'] =sum([blocks[f_id]['filters'] for f_id in layers])
            block['from'] = layers
            block['width'] = blocks[layers[0]]['width']
            block['height'] = blocks[layers[0]]['width']
            need_save.extend(layers)

        elif block['type'] == 'yolo' or block['type'] == 'my_yolo':
            print('%5d %-6s' % (ind, block['type']))
            width = blocks[ind-1]['width']
            height = blocks[ind-1]['height']
            block['width'] = width
            block['height'] = height
            block['in_channel'] = blocks[ind - 1]['filters']
            block['stride'] = blocks[0]['width']//width
            blocks[0]['stride'].append(block['stride'])

        elif block['type'] == 'shortcut':
            from_id = int(block['from'])
            from_id = from_id if from_id > 0 else from_id+ind
            block['filters'] = blocks[from_id]['filters']
            block['from'] = from_id
            block['width'] = blocks[from_id]['width']
            block['height'] = blocks[from_id]['height']
            block['in_channel'] = blocks[from_id]['filters']
            print('%5d %-6s %d' % (ind, 'shortcut', from_id))
            need_save.append(from_id)

        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            width = blocks[ind-1]['width']*stride
            height = blocks[ind-1]['height']*stride
            block['width'] = width
            block['height'] = height
            block['filters'] = blocks[ind - 1]['filters']
            block['in_channel'] = blocks[ind - 1]['filters']
            print('%5d %-6s            %dx   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'upsample', stride, blocks[ind-1]['width'], blocks[ind-1]['height'], blocks[ind-1]['filters'], width, height, filters))
        else:
            print('unknown type %s' % (block['type']))

    return blocks,need_save
if __name__ == '__main__':
    ifnfo = parse_cfg('yolo_new.cfg')
    pass