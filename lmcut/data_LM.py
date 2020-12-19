# credit: the code below is modified from https://github.com/m-hahn/tabula-rasa-rnns
import logging

import torch

cuda = torch.cuda.is_available()

# define vocabulary
thai_chars = [
    u'\n', u' ', u'!', u'"', u'#', u'$', u'%', u'&', "'", u'(', u')', u'*', u'+',
    u',', u'-', u'.', u'/', u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8',
    u'9', u':', u';', u'<', u'=', u'>', u'?', u'@', u'A', u'B', u'C', u'D', u'E',
    u'F', u'G', u'H', u'I', u'J', u'K', u'L', u'M', u'N', u'O', u'P', u'Q', u'R',
    u'S', u'T', u'U', u'V', u'W', u'X', u'Y', u'Z', u'[', u'\\', u']', u'^', u'_',
    u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm',
    u'n', u'o', u'other', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y',
    u'z', u'}', u'~', u'ก', u'ข', u'ฃ', u'ค', u'ฅ', u'ฆ', u'ง', u'จ', u'ฉ', u'ช',
    u'ซ', u'ฌ', u'ญ', u'ฎ', u'ฏ', u'ฐ', u'ฑ', u'ฒ', u'ณ', u'ด', u'ต', u'ถ', u'ท',
    u'ธ', u'น', u'บ', u'ป', u'ผ', u'ฝ', u'พ', u'ฟ', u'ภ', u'ม', u'ย', u'ร', u'ฤ',
    u'ล', u'ว', u'ศ', u'ษ', u'ส', u'ห', u'ฬ', u'อ', u'ฮ', u'ฯ', u'ะ', u'ั', u'า',
    u'ำ', u'ิ', u'ี', u'ึ', u'ื', u'ุ', u'ู', u'ฺ', u'เ', u'แ', u'โ', u'ใ', u'ไ',
    u'ๅ', u'ๆ', u'็', u'่', u'้', u'๊', u'๋', u'์', u'ํ', u'๐', u'๑', u'๒', u'๓',
    u'๔', u'๕', u'๖', u'๗', u'๘', u'๙', u'‘', u'’', u'\ufeff'
]
chars = thai_chars #+ "."
itos = ['<START>'] + ['<END>'] + ['<UNK>'] + list(chars)
stoi = {v: k for k, v in enumerate(itos)}


# create batches
def _prepareDatasetChunks(args, data):
    count = 0
    logging.info("Prepare chunks")
    numerified = []
    for chunk in data:
        for char in chunk:
            if char == " " or char == "":
                continue
            count += 1
            numerified.append(stoi[char] if char in stoi else 2)

        cutoff = int(len(numerified) / (args.batchSize * args.sequence_length)) \
                 * (args.batchSize * args.sequence_length)

        numerifiedCurrent = numerified[:cutoff]
        numerified = numerified[cutoff:]
        if cuda:
            numerifiedCurrent = torch.LongTensor(numerifiedCurrent) \
                .view(args.batchSize, -1, args.sequence_length) \
                .transpose(0, 1).transpose(1, 2).cuda()
        else:
            numerifiedCurrent = torch.LongTensor(numerifiedCurrent) \
                .view(args.batchSize, -1, args.sequence_length) \
                .transpose(0, 1).transpose(1, 2)

        numberOfSequences = numerifiedCurrent.size()[0]
        for i in range(numberOfSequences):
            yield numerifiedCurrent[i]
