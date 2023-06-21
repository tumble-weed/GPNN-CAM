#===========================================================
import sys
import ipdb

def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print()
        # ...then start the debugger in post-mortem mode.
        ipdb.post_mortem(tb)

sys.excepthook = info
#===========================================================

if False:
    #===========================================================
    import sys
    import ipdb

    # import sys
    # def my_except_hook(exctype, value, traceback):
    #     if exctype == KeyboardInterrupt:
    #         print "Handler code goes here"
    #     else:
    #         sys.__excepthook__(exctype, value, traceback)
    # sys.excepthook = my_except_hook

    def info(type, value, tb):
        if hasattr(sys, 'ps1') or not sys.stderr.isatty():
            # we are in interactive mode or we don't have a tty-like
            # device, so we call the default hook
            sys.__excepthook__(type, value, tb)
        else:
            import traceback
            
            # we are NOT in interactive mode, print the exception...
            traceback.print_exception(type, value, tb)
            
            # print()        
            # print(type)
            # print(value)
            # if type == FileNotFoundError:
            #     print("Handler code goes here")
            #     import ipdb;ipdb.set_trace()
            # else:
            #     sys.__excepthook__(exctype, value, traceback)


            # ...then start the debugger in post-mortem mode.
            ipdb.post_mortem(tb)

    sys.excepthook = info
    #===========================================================
