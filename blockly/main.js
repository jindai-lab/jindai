var _blocks_dict = {};
var _id = location.href.split('/').pop();
var _task = {};

function auto_parse(val) {
  if (typeof val !== 'string') return val;
  if (val === 'FALSE' || val === 'TRUE') return val === 'TRUE';
  if (val.match(/^[+-]?\d+\.?\d*[Ee]?[+-]?\d*$/)) return +val;
  return val;
}

function save() {
  var xml = Blockly.Xml.workspaceToDom(Blockly.mainWorkspace);
  if (!xml.children.length) return;

  function from_xml(block) {
    var j = []
    while (block) {
      // iterative through 'next' tags to get a sequence of blocks
      var next = null, obj = {
        name: block.getAttribute('type'),
        args: {}
      }
      for (var child of block.children) {
        switch (child.tagName) {
          case 'next':
            next = child.children[0];
            break;
          case 'field':
            obj.args[child.getAttribute('name')] = auto_parse(child.innerHTML);
            break;
          case 'value':
            obj.args[child.getAttribute('name')] = auto_parse(child.getElementsByTagName('field')[0].innerHTML);
            break;
          case 'statement':
            obj.args[child.getAttribute('name')] = from_xml(child.children[0]);
            break;
        }
      }
      j.push([obj.name, obj.args])
      block = next;
    }
    return j
  }

  var block = xml.children[0];
  var j = from_xml(block)
  
  j = {
    datasource: j[0][0],
    datasource_config: j[0][1],
    pipeline: j.slice(1)
  }
  console.log(j);
  
  Object.assign(_task, j);
  $.ajax({
    url: '/api/tasks/' + _id,
    type: 'POST',
    data: JSON.stringify(_task),
    contentType: 'application/json',
    success: function () {
      location.href = '/tasks/' + _id;
    }
  });
}

function load() {
  $.when($.get('/api/help/pipelines'), $.get('/api/help/datasources'), $.get('/api/tasks')).done((pls, dss, tss) => {
    var toolbox = {
      kind: "categoryToolbox",
      contents: []
    }
    pls = pls[0].result; dss = dss[0].result; tss = tss[0].result;
    [[dss, 'ds'], [pls, 'pl']].forEach(group_object => {
      const atype = group_object[1]
      toolbox.contents.push({
        kind: 'category',
        name: atype == 'ds' ? '数据源' : '过程处理',
        contents: []
      })
      for (var group in group_object[0]) {
        toolbox.contents.push({
          kind: 'category',
          name: group,
          colour: atype == 'ds' ? 220 : 160,
          contents: []
        })
        let toolbox_contents = toolbox.contents.slice(-1)[0].contents
        let items = group_object[0][group]
        for (var item_key in items) {
          let item = items[item_key]
          let obj = {
            previousStatement: atype == 'ds' ? undefined : null,
            nextStatement: null,
            colour: atype == 'ds' ? 220 : 160,
            message0: `${item.name} ${item.doc}`,
          }
          _blocks_dict[item.name] = item;
          for (var i = 0; i < item.args.length; ++i) {
            let arg = item.args[i]
            arg.description = arg.description || arg.name
            obj[`message${i + 1}`] = arg.description.includes('%1') ? arg.description : `${arg.description}：%1`
            obj[`args${i + 1}`] = [arg.type.includes('|') ? {
              type: 'field_dropdown',
              name: arg.name,
              options: arg.type.split('|').map(x => x.includes(':') ? x.split(':') : [x, x])
            } : ['float', 'int'].includes(arg.type) ? {
              type: 'field_number',
              name: arg.name
            } : 'bool' == arg.type ? {
              type: 'field_checkbox',
              name: arg.name
            } : 'TASK' == arg.type ? {
              type: 'field_dropdown',
              name: arg.name,
              options: tss.map(x => [x.name, x._id])
            } : 'pipeline' == arg.type ? {
              type: 'input_statement',
              name: arg.name
            } : {
              type: 'input_value',
              name: arg.name
            }]
          }
          Blockly.Blocks[item.name] = {
            init: function () {
              this.jsonInit(obj)
            }
          }
          toolbox_contents.push({
            kind: 'block',
            type: item.name
          })
        }
      }
    })
    toolbox.contents.push({
      kind: 'category',
      name: '常量',
      contents: [
        { kind: 'block', type: 'text' },
        { kind: 'block', type: 'text_multiline' },
        { kind: 'block', type: 'math_number' },
      ]
    })
    console.log('toolbox', toolbox)
    Blockly.inject('blocklyDiv', {
      toolbox: toolbox,
      scrollbars: false,
      trashcan: true
    });
    $.get('/api/tasks/' + _id).then(data => {
      data = data.result;
      _task = data;
      let workspace = Blockly.getMainWorkspace();
      workspace.clear();

      function to_xml(parent, tuples) {
        var index = 0;

        for (var tup of tuples) {
          if (index > 0) {
            var next = document.createElement('next');
            parent.appendChild(next);
            parent = next;
          }

          let name = tup[0], args = tup[1];
          var block = document.createElement('block');
          block.setAttribute('type', name);
          if (!_blocks_dict[name]) {
            console.log('?unknown block', name);
            continue;
          }
          for (var argname in args) {
            let argvalue = args[argname], arg = _blocks_dict[name].args.filter(x => x.name == argname)[0];
            if (!arg) {
              console.log('?arg', argname);
              continue;
            }
            let argtype = arg.type;
            if (['bool', 'float', 'int', 'TASK'].includes(argtype) || argtype.includes('|')) {
              var field = document.createElement('field');
              field.setAttribute('name', argname);
              argvalue = '' + argvalue;
              if (argtype == 'bool') argvalue = argvalue.toUpperCase();
              field.innerHTML = argvalue;
              block.appendChild(field);
            } else if (argtype == 'pipeline') {
              var field = document.createElement('statement');
              field.setAttribute('name', argname);
              to_xml(field, argvalue);
              block.appendChild(field);
            } else {
              var valuenode = document.createElement('value');
              valuenode.setAttribute('name', argname);
              var valueblock = document.createElement('block');
              valueblock.setAttribute('type', argvalue.includes('\n') ? 'text_multiline' : 'text');
              var valueblockfield = document.createElement('field');
              valueblockfield.setAttribute('name', 'TEXT');
              valueblockfield.innerHTML = argvalue;
              valueblock.appendChild(valueblockfield);
              valuenode.appendChild(valueblock);
              block.appendChild(valuenode);
            }
          }

          parent.append(block);
          parent = block;
          ++index;
        }
      }

      if (data) {
        var xml = document.createElement('xml');
        to_xml(xml, [[data.datasource, data.datasource_config], ...data.pipeline]);
        console.log(xml);
        Blockly.Xml.domToWorkspace(xml, workspace);
      }
    });
  })
}

load();