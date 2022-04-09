import re 
import torch 


def mt_convert_url(url):
    url = url.replace('https', 'http', 1)
    if 'download-image.sankuai.com' in url:
        return url
    elif not url.startswith('http'):
        # 是上海UGC的短码url, 先转化为内网url
        return 'http://download-image.sankuai.com/ugcpic/' + url
    elif 'meituan.net' in url:
        pat = re.compile("[^/]+.meituan.net")
        return re.sub(pat, "download-image.sankuai.com", url)
    elif 'p.vip.sankuai.com' in url:
        pat = re.compile("p.vip.sankuai.com")
        return re.sub(pat, "download-image.sankuai.com", url)
    elif 'mtmos.com' in url:
        pat = re.compile("mtmos.com")
        return re.sub(pat, "mss.vip.sankuai.com", url)
    elif 'mss.sankuai.com' in url:
        pat = re.compile("mss.sankuai.com")
        return re.sub(pat, "mss.vip.sankuai.com", url)
    elif 's3plus.sankuai.com' in url:
        pat = re.compile("s3plus.sankuai.com")
        return re.sub(pat, "s3plus.vip.sankuai.com", url)
    elif 'mss-shon.sankuai.com' in url:
        pat = re.compile("mss-shon.sankuai.com")
        return re.sub(pat, "mss-shon.vip.sankuai.com", url)
    elif 'sankuai.com' in url:
        pat = re.compile("[^/]+.sankuai.com")
        return re.sub(pat, "download-image.sankuai.com", url)
    else:
        return url  


def accuracy_compute(logits, labels, top_k=5): 
    bsz, seq_len, _ = logits.size()
    logits = logits.contiguous().view(bsz*seq_len, -1)
    _, idx = torch.topk(logits, top_k, -1) 
    correct = idx.eq(labels.view(-1, 1).expand_as(idx)) 
    correct_total = correct.view(-1).float().sum().item()
    nums = labels.view(-1).detach().cpu().numpy()
    length = 0 
    for num in nums:
        if num != 0:
            length += 1
    return correct_total / float(length) 


