export interface User {
  id: number;
  full_name: string;
  email: string;
  created_at: string;
  updated_at: string;
}

export interface Area {
  id: number;
  name: string;
  color: string;
  user_id: number;
  created_at: string;
  updated_at: string;
}

export enum Priority {
  High = 'High',
  Medium = 'Medium',
  Low = 'Low',
}

export interface Task {
  id: any;
  title: string;
  description?: string;
  area_id?: number | null;
  priority?: Priority;
  completed: boolean;
  due_date?: string;
  user_id?: number;
  created_at: string;
  updated_at: string;
}

export interface Note {
  id: any;
  title: string;
  content: string;
  area_id?: number | null;
  user_id?: number;
  created_at: string;
  updated_at: string;
}

export interface AppData {
  user: User;
  tasks: Task[];
  notes: Note[];
  areas: Area[];
}

export interface TaskSearchResult {
  id: number;
  title: string;
  description?: string;
  priority?: Priority;
  due_date?: string;
  type: "task";
}

export interface NoteSearchResult {
  id: number;
  title: string;
  content: string;
  type: "note";
}

export type SearchResult = TaskSearchResult | NoteSearchResult;