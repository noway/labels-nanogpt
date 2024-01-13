from datetime import datetime

date1_str = '2024-01-12 06:12:40.135382'
date2_str = '2024-01-12 06:21:53.841030'

date_format = '%Y-%m-%d %H:%M:%S.%f'

date1 = datetime.strptime(date1_str, date_format)
date2 = datetime.strptime(date2_str, date_format)

difference = date2 - date1
difference_in_seconds = difference.total_seconds()

needed_training_seconds = difference_in_seconds / 1000 * (5000 * 4 * 2)

hours, remainder = divmod(needed_training_seconds, 3600)
minutes, seconds = divmod(remainder, 60)

print('hours', hours)
print('minutes', minutes)
print('seconds', seconds)
