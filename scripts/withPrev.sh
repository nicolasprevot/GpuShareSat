 jq 'reduce .[] as $item ([]; . + [{p: .[-1].c, c: $item}]) | .[1:]'
