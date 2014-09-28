# -*- coding: utf-8 -*-


class HTML(object):
    @classmethod
    def h1(self, txt, id=None):
        if id is not None:
            st = '<h1 id="%s">%s</h1>\n\n' % (id, txt)
        else:
            st = '<h1>%s</h1>\n\n' % txt
        return st

    @classmethod
    def h3(self, txt, id=None):
        if id is not None:
            st = '<h3 id="%s">%s</h3>\n\n' % (id, txt)
        else:
            st = '<h3>%s</h3>\n\n' % txt
        return st

    @classmethod
    def hr(self):
        return '<hr>\n\n'

    @classmethod
    def table(self, tbl, row_color, link=None):
        st = '<table border="1" cellspacing="0" cellpadding="5">\n'
        # header
        st += '<tr>\n'
        for val in tbl[0]:
            st += '<th><font color=%s>%s</font></th>\n' % (row_color[0], val)
        st += '</tr>\n'
        # contents
        for row in xrange(1, len(tbl)):
            st += '<tr>\n'
            for col, val in enumerate(tbl[row]):
                if link is not None and link[row] is not None and col == 0:
                    st += '<td><font color=%s><a href="%s">%s</a></font></td>\n' %\
                        (row_color[row], link[row], val)
                else:
                    st += '<td><font color=%s>%s</font></td>\n' %\
                        (row_color[row], val)
            st += '</tr>\n'
        st += '</table>\n\n'
        return st

    @classmethod
    def list_item(self, lst):
        st = '<p>\n<ul>\n'
        for line in lst:
            st += '<li>%s</li>\n' % line
        st += '</ul>\n</p>\n\n'
        return st

    @classmethod
    def pre(self, txt):
        return '<pre>\n%s\n</pre>\n\n' % txt

    @classmethod
    def img(self, path, height, alt):
        return '<img border="0" src="%s" height="%d" alt="%s">\n\n' %\
            (path, height, alt)

    @classmethod
    def list_item_with_title(self, lst):
        st = '<p>\n'
        for line in lst:
            st += '<strong>%s</strong>\n<ul>\n' % line[0]
            for val in line[1]:
                st += '<li>%s</li>\n' % val
            st += '</ul>\n'
        st += '</p>\n\n'
        return st

    @classmethod
    def open(self, path, mode):
        fo = open(path, mode)
        fo.write('<html>\n\n')
        fo.write('<head>\n')
        fo.write('<meta http-equiv="Content-Type" content="text/html;' +
                 'charset=utf-8">\n')
        fo.write('<title>Analysis report</title>')
        fo.write('</head>\n\n')
        fo.write('<body>\n')
        return fo

    @classmethod
    def close(self, fo):
        fo.write('</body>\n</html>\n')
        fo.close()
