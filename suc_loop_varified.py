_suc_count = 0
maxcount = 10
oneloop_done_count = 10
suc_item = 1
curr_count = 0

while True:
    _suc_count += 1
    if _suc_count >= suc_item:
        curr_count += 1
        if curr_count > oneloop_done_count:
            curr_count = 0
            _suc_count = 0
            suc_item += 1
    else:
        print('done false')
