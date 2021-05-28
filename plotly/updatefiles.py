import time

start = time.time()

exec(open('cleandata.py').read())
exec(open('timeline.py').read())
exec(open('timelinebar.py').read())
exec(open('waithistogram.py').read())
exec(open('waitbar.py').read())
exec(open('splitters.py').read())
exec(open('poolscatter.py').read())

print('\nAll files updated. Time to complete: %.2f minutes.' % ((time.time() - start)/60))
