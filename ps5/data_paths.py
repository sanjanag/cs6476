actions = ['botharms', 'crouch', 'leftarmup', 'punch', 'rightkick']
seqpaths = ['./PS5_Data/botharms/botharms-up-p1-1/',
            './PS5_Data/botharms/botharms-up-p1-2/',
            './PS5_Data/botharms/botharms-up-p2-1/',
            './PS5_Data/botharms/botharms-up-p2-2/',

            './PS5_Data/crouch/crouch-p1-1/',
            './PS5_Data/crouch/crouch-p1-2/',
            './PS5_Data/crouch/crouch-p2-1/',
            './PS5_Data/crouch/crouch-p2-2/',

            './PS5_Data/leftarmup/leftarm-up-p1-1/',
            './PS5_Data/leftarmup/leftarm-up-p1-2/',
            './PS5_Data/leftarmup/leftarm-up-p2-1/',
            './PS5_Data/leftarmup/leftarm-up-p2-2/',

            './PS5_Data/punch/punch-p1-1/',
            './PS5_Data/punch/punch-p1-2/',
            './PS5_Data/punch/punch-p2-1/',
            './PS5_Data/punch/punch-p2-2/',

            './PS5_Data/rightkick/rightkick-p1-1/',
            './PS5_Data/rightkick/rightkick-p1-2/',
            './PS5_Data/rightkick/rightkick-p2-1/',
            './PS5_Data/rightkick/rightkick-p2-2/']
allLabels = [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 4
actions_dict = {i:actions[i-1] for i in range(1,6)}
# print(actions_dict)
# print(allLabels)
